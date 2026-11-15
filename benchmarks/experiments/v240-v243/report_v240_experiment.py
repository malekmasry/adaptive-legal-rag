"""Read-only analysis of the frozen experiment. No model requests or selection."""
import argparse,copy,hashlib,html,json,statistics,zipfile
from collections import Counter
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W/'v240-experiment'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def cap(name,ident):
    if name=='v239_development':name='v239_development_recovered'
    return read(ROOT/'generated'/name/'captured'/(ident+'.json'))

def analyze(phase):
    if phase=='holdout':assert (ROOT/'selection.json').exists()
    summary=read(ROOT/(phase+'-summary.json'));covers=read(ROOT/(phase+'-coverage.json'))
    cases=read(ROOT/(phase+'.json'));answers=read(ROOT/(phase+'-answers.json'))
    judgments=read(ROOT/(phase+'-judgments.json'));baseline=answers['v239_'+phase]
    systems={}
    for name,rows in covers.items():
        captures={c['id']:cap(name,c['id']) for c in cases}
        anchors=[a for r in rows for a in r.get('anchors',[])]
        context_changes=[];whole_payload_changes=[];metadata_only=[];topic_order_only=[];degraded={}
        for c in cases:
            ident=c['id'];value=captures[ident];reference=cap('v239_'+phase,ident)
            p=value['calls'][0]['payload'];bp=reference['calls'][0]['payload']
            if p['evidence']!=bp['evidence']:context_changes.append(ident)
            if p!=bp:whole_payload_changes.append(ident)
            changed=[k for k in p if p[k]!=bp.get(k)]
            if changed==['context_limits']:metadata_only.append(ident)
            def normalize_topics(value):
                value=copy.deepcopy(value)
                for item in value.get('context_limits',{}).get('decision_search',{}).get('selected',[]):
                    if 'topics' in item:item['topics']=sorted(item['topics'])
                return value
            if p!=bp and normalize_topics(p)==normalize_topics(bp):topic_order_only.append(ident)
            result=value.get('result') or {};trace=result.get('trace',{})
            if trace.get('retrieval',{}).get('degraded'):degraded[ident]=trace['retrieval']['degraded']
            assert p['context_limits']['total_evidence_tokens']<=p['context_limits'].get('configured_context_budget',18000)
        systems[name]={
            'answer_count':len(answers[name]),'status_counts':dict(Counter((v.get('result') or {}).get('status') for v in captures.values())),
            'all_required_anchors_cases':sum(r.get('all_anchors_present') is True for r in rows),
            'cases_with_anchors':sum(r.get('all_anchors_present') is not None for r in rows),
            'present_anchors':sum(a['present'] for a in anchors),'required_anchors':len(anchors),
            'missing_anchor_ids':[r['id'] for r in rows if r.get('all_anchors_present') is False],
            'ready_draft_ids':[r['id'] for r in rows if r.get('draft_origin')=='source_rules'],
            'review_status_counts':dict(Counter(r.get('review') for r in rows)),
            'evidence_changed_vs_v239':context_changes,'first_payload_changed_vs_v239':whole_payload_changes,
            'only_context_metadata_changed_vs_v239':metadata_only,'only_unordered_topic_labels_changed_vs_v239':topic_order_only,
            'answer_changed_vs_v239':[c['id'] for c in cases if answers[name][c['id']]!=baseline[c['id']]],
            'omitted_document_ids':[r['id'] for r in rows if r.get('omitted_documents')],
            'retrieval_degraded':degraded,'context_budget_violations':0,
            'evidence_tokens_mean':statistics.mean(r['evidence_tokens'] for r in rows),
            'evidence_tokens_max':max(r['evidence_tokens'] for r in rows),
            'logical_model_calls_including_retained_failed_review_attempts':sum(len(v['calls']) for v in captures.values()),
            'runtime_seconds_with_shared_caches':summary['runtime'][name]['seconds']}
    judge_audit={}
    for name in dict.fromkeys(r['system'] for r in judgments):
        rows=[r for r in judgments if r['system']==name]
        judge_audit[name]={'metric':summary['systems'][name],
            'wins':[r['id'] for r in rows if r.get('score') is not None and r['score']>.5],
            'losses':[r['id'] for r in rows if r.get('score') is not None and r['score']<.5],
            'ties':[r['id'] for r in rows if r.get('score')==.5],
            'errors':[r['id'] for r in rows if r['status']!='ok'],
            'fabrication_zero_ids':[r['id'] for r in rows if r.get('fabrication_zero')],
            'order_disagreement_ids':[r['id'] for r in rows if r.get('order_gap',0)>0],
            'fabrication_disagreement_ids':[r['id'] for r in rows if r.get('fabrication_disagreement')]}
    return {'phase':phase,'systems':systems,'judge_audit':judge_audit,'wall_seconds':summary['seconds'],
        'summary_sha256':sha(ROOT/(phase+'-summary.json'))}

def render(report):
    e=lambda x:html.escape(str(x));parts=['<!doctype html><meta charset="utf-8"><title>PSB: v239 / v240 / v241</title>',
        '<style>body{font:16px/1.55 system-ui;max-width:1150px;margin:36px auto;padding:0 20px;color:#152233}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #c9d3df;padding:8px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;font:inherit;background:#f3f6f9;padding:16px}summary{cursor:pointer;font-weight:600;margin:14px 0}.note{background:#fff4d5;padding:16px}</style>',
        '<h1>Диагностика RAG: v239, v240, v241</h1>',
        '<p class="note">Локальное попарное сравнение с новым baseline v239. 50 означает сопоставимое качество. Эти числа нельзя сравнивать с прежним тестом из 20 вопросов или баллом платформы. Судья — наша воспроизводимая гипотеза на DeepSeek, не раскрытый промпт ПСБ.</p>',
        '<p>v240: изменение упаковки контекста при переполнении. v241: отдельное отключение программных готовых черновиков, без v240. Oracle: вручную выбранные исходные источники; диагностический вариант, не сабмит.</p>']
    if 'selection' in report:
        parts+=['<h2>Выбор до holdout</h2><pre>'+e(json.dumps(report['selection'],ensure_ascii=False,indent=2))+'</pre>']
    for phase,audit in report['phases'].items():
        parts+=['<h2>'+e(phase)+'</h2><table><tr><th>Сравнение</th><th>Баллы</th><th>Match Rate</th><th>Win Rate</th><th>Победы / поражения / ничьи</th><th>Покрытие</th></tr>']
        for name,item in audit['judge_audit'].items():
            m=item['metric'];parts+=['<tr>'+''.join('<td>'+e(v)+'</td>' for v in [name,round(m['score'],4) if m['score'] is not None else None,m.get('match_rate'),m.get('win_rate'),f"{m.get('wins')}/{m.get('losses')}/{m.get('ties')}",f"{m['evaluated']}/{m['attempted']}"])+'</tr>']
        parts+=['</table><details><summary>Контекст, вызовы и техническая диагностика</summary><pre>'+e(json.dumps(audit['systems'],ensure_ascii=False,indent=2))+'</pre></details>']
        cases=read(ROOT/(phase+'.json'));answers=read(ROOT/(phase+'-answers.json'))
        judgments=read(ROOT/(phase+'-judgments.json'))
        for c in cases:
            parts+=['<details><summary>'+e(c['id']+' — '+c['question'])+'</summary><b>Эталон</b><pre>'+e(c['golden'])+'</pre>']
            for name,rows in answers.items():parts+=['<b>'+e(name)+'</b><pre>'+e(rows[c['id']])+'</pre>']
            parts+=['<details><summary>Два раунда судейства</summary><pre>'+e(json.dumps([r for r in judgments if r['id']==c['id']],ensure_ascii=False,indent=2))+'</pre></details></details>']
    if (ROOT/'manual-audit.json').exists():parts+=['<h2>Ручная проверка</h2><pre>'+e(json.dumps(read(ROOT/'manual-audit.json'),ensure_ascii=False,indent=2))+'</pre>']
    parts+=['<h2>Расход</h2><pre>'+e(json.dumps(report['usage'],ensure_ascii=False,indent=2))+'</pre>',
        '<p>Время вариантов измерено с общим кешем точных запросов. Baseline development восстановлен после HTTP 402: девять завершённых ответов переиспользованы, два продолжены с проверки, семь выполнены отдельно. Время не является сравнением скорости холодных запусков.</p>']
    return '\n'.join(parts)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--holdout',action='store_true');a=parser.parse_args()
    phases=['development']+(['holdout'] if a.holdout else [])
    report={'phases':{p:analyze(p) for p in phases}}
    if (ROOT/'selection.json').exists():report['selection']=read(ROOT/'selection.json')
    done=[r for r in map(json.loads,(ROOT/'calls.jsonl').read_text(encoding='utf8').splitlines()) if r['event']=='completed']
    report['usage']={}
    for role in ('GENERATOR','JUDGE','RERANKER'):
        rows=[r for r in done if r['role']==role]
        report['usage'][role]={'calls':len(rows),'successful':sum(r['status']=='ok' for r in rows),
            'tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in rows),
            'rerank_input_tokens':sum(r.get('meta',{}).get('tokens',{}).get('input_tokens',0) for r in rows)}
        expected='Qwen/Qwen3-Reranker-8B' if role=='RERANKER' else 'deepseek-v4.1-flash'
        assert all(r['model']==expected for r in rows)
    report['usage']['api_errors']=sum(r['status']!='ok' for r in done)
    frozen=read(ROOT/'runtime-freeze.json')
    for name,meta in frozen['packages'].items():
        assert sha(ROOT/'packages'/name)==meta['sha256']
        with zipfile.ZipFile(ROOT/'packages'/name) as z:assert z.testzip() is None
    report['verified_package_sha256']={n:m['sha256'] for n,m in frozen['packages'].items()}
    write(ROOT/'analysis.json',report);(ROOT/'REPORT.html').write_text(render(report),encoding='utf8')
    print(json.dumps({'phases':phases,'usage':report['usage'],'packages_verified':True}))
