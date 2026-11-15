"""Final read-only report for the all-three-features experiment."""
import hashlib,html,json,statistics,zipfile
from collections import Counter
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W/'v242-experiment';OLD=W/'v240-experiment'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def usage(root):
    done=[r for r in map(json.loads,(root/'calls.jsonl').read_text(encoding='utf8').splitlines()) if r['event']=='completed']
    out={}
    for role in ['GENERATOR','JUDGE','RERANKER']:
        rows=[r for r in done if r['role']==role]
        expected='Qwen/Qwen3-Reranker-8B' if role=='RERANKER' else 'deepseek-v4.1-flash'
        assert all(r['model']==expected for r in rows)
        out[role]={'calls':len(rows),'successful_calls':sum(r['status']=='ok' for r in rows),
            'tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in rows),
            'rerank_input_tokens':sum(r.get('meta',{}).get('tokens',{}).get('input_tokens',0) for r in rows)}
    out['api_errors']=sum(r['status']!='ok' for r in done);return out

report={'selected':read(ROOT/'selection.json'),'prototype':read(ROOT/'development-summary.json'),
    'phases':{},'usage':usage(ROOT),'previous_experiment_usage':usage(OLD),'manual_audit':read(ROOT/'manual-audit.json')}
report['final_rendering_verification']=read(ROOT/'final-rendering-verification.json')
for phase in ['development','holdout']:
    summary=read(ROOT/('v243-'+phase+'-summary.json'));cases=read(ROOT/(phase+'.json'))
    answers=read(ROOT/('v243-'+phase+'-answers.json'));rows=read(ROOT/('v243-'+phase+'-judgments.json'))
    captures={c['id']:read(ROOT/'generated'/('v242_development' if phase=='development' else 'v243_holdout')/'captured'/(c['id']+'.json')) for c in cases}
    traces=[v['result']['trace'] for v in captures.values()]
    assert all(t['budget']['total_evidence_tokens']<=18000 for t in traces)
    coverage=read(ROOT/('development-coverage.json' if phase=='development' else 'v243-holdout-coverage.json'))
    anchors=[a for row in coverage for a in row.get('anchors',[])]
    report['phases'][phase]={'aggregate':summary['aggregate'],
        'runtime':read(ROOT/'development-runtime.json') if phase=='development' else summary['runtime'],
        'full_question_brief_count':sum(t.get('question_brief',{}).get('status')=='applied' for t in traces),
        'brief_statuses':dict(Counter(t.get('question_brief',{}).get('status') for t in traces)),
        'required_parts':{c['id']:captures[c['id']]['result']['trace'].get('question_brief',{}).get('parts') for c in cases},
        'schema_fallback_ids':[c['id'] for c in cases if captures[c['id']]['result']['trace'].get('question_brief',{}).get('schema_fallback')],
        'retrieval_degraded_ids':[c['id'] for c in cases if captures[c['id']]['result']['trace']['retrieval'].get('degraded')],
        'present_anchors':sum(a['present'] for a in anchors),'required_anchors':len(anchors),
        'context_budget_violations':0,'context_tokens_mean':statistics.mean(t['budget']['total_evidence_tokens'] for t in traces),
        'wins':[r['id'] for r in rows if r.get('score') is not None and r['score']>.5],
        'losses':[r['id'] for r in rows if r.get('score') is not None and r['score']<.5],
        'fabrication_zero_ids':[r['id'] for r in rows if r.get('fabrication_zero')],
        'judge_error_ids':[r['id'] for r in rows if r['status']!='ok'],
        'order_disagreement_ids':[r['id'] for r in rows if r.get('order_gap',0)>0],
        'mean_answer_characters':{name:statistics.mean(len(text) for text in values.values()) for name,values in answers.items()}}
report['verified_package_sha256']={}
for name,expected in [('v242.zip',read(ROOT/'source-freeze.json')['package_sha256']),('v243.zip',read(ROOT/'v243-source-freeze.json')['package_sha256']),('v243-final.zip',report['final_rendering_verification']['package_sha256'])]:
    assert sha(ROOT/'packages'/name)==expected
    with zipfile.ZipFile(ROOT/'packages'/name) as z:assert z.testzip() is None
    report['verified_package_sha256'][name]=expected
write(ROOT/'analysis.json',report)
e=lambda v:html.escape(str(v));parts=['<!doctype html><meta charset="utf-8"><title>PSB v243: три изменения</title>',
 '<style>body{font:16px/1.55 system-ui;max-width:1150px;margin:36px auto;padding:0 20px;color:#182536}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #c9d3df;padding:8px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;background:#f3f6f9;padding:16px;font:inherit}summary{font-weight:600;cursor:pointer;margin:12px 0}.note{padding:16px;background:#fff2cf}</style>',
 '<h1>v243: обязательные части, точные ссылки, памятка по условиям</h1>',
 '<p class="note">Сравнение с замороженными ответами v239. 50 баллов означает ничью, а не 50% правильных ответов. Это локальный судья DeepSeek; результат не является оценкой или прогнозом платформы ПСБ.</p>',
 '<p>Добавлен отдельный вызов DeepSeek перед генерацией. Он выделяет части конкретного вопроса и связывает условия с источниками. Схема ответа требует покрытия всех выделенных частей. Каталог ссылок копирует положения из текущих источников. Правовая база и поиск сохранены. При сбое дополнительного этапа остаётся исходный путь ответа.</p>',
 '<p>v242 — исходная сборка всех трёх изменений. v243 исправляет только утечку служебных меток P1/P2 в текст. На development проверено точное воспроизведение всех 18 ответов; изменились 5, генерация не повторялась. На holdout выполнен полный живой прогон v243. 82 теста прошли.</p>',
 '<p>Итоговый ZIP v243-final дополнительно переводит внутреннее «в evidence» в «в предоставленных источниках». Прошли 83 теста и точное повторное оформление всех 30 сохранённых ответов; изменился только H12, ссылки и факты сохранены. Новый полный живой прогон и переоценка этого одного изменения формулировки не проводились. Числа ниже относятся к замороженному v243 до этой косметики.</p>',
 '<h2>Результаты</h2><table><tr><th>Выборка</th><th>Баллы</th><th>Победы / поражения / ничьи</th><th>Match Rate</th><th>Win Rate</th><th>Время продукта</th></tr>']
for phase,value in report['phases'].items():
    m=value['aggregate'];parts+=['<tr>'+''.join('<td>'+e(v)+'</td>' for v in [phase,round(m['score'],4),f"{m['wins']}/{m['losses']}/{m['ties']}",m['match_rate'],m['win_rate'],str(value['runtime']['seconds'])+' с'])+'</tr>']
parts+=['</table><p>Времена измерены на 3 параллельных вопросах, с общими кешами BGE/Qwen. Это не оценка времени полного скрытого прогона ПСБ. Для development указано время исходного полного v242, а не быстрого повторного оформления.</p>',
 '<h2>Выбор до holdout</h2><pre>'+e(json.dumps(report['selected'],ensure_ascii=False,indent=2))+'</pre>',
 '<h2>Ручной разбор</h2><pre>'+e(json.dumps(report['manual_audit'],ensure_ascii=False,indent=2))+'</pre>']
for phase in ['development','holdout']:
    parts+=['<h2>'+e(phase)+': все ответы</h2>']
    answers=read(ROOT/('v243-'+phase+'-answers.json'));rows=read(ROOT/('v243-'+phase+'-judgments.json'))
    for c in read(ROOT/(phase+'.json')):
        parts+=['<details><summary>'+e(c['id']+' — '+c['question'])+'</summary><b>Эталон</b><pre>'+e(c['golden'])+'</pre>']
        for name,data in answers.items():parts+=['<b>'+e(name)+'</b><pre>'+e(data[c['id']])+'</pre>']
        final=read(ROOT/'final-answers.json')[phase][c['id']]
        if final!=answers['v243_'+phase][c['id']]:parts+=['<b>Итоговая формулировка v243-final (без повторного судейства)</b><pre>'+e(final)+'</pre>']
        parts+=['<details><summary>Два раунда судейства</summary><pre>'+e(json.dumps(next(r for r in rows if r['id']==c['id']),ensure_ascii=False,indent=2))+'</pre></details></details>']
parts+=['<h2>Расход этого этапа</h2><pre>'+e(json.dumps(report['usage'],ensure_ascii=False,indent=2))+'</pre>',
 '<h2>Контекст и техническая диагностика</h2><pre>'+e(json.dumps(report['phases'],ensure_ascii=False,indent=2))+'</pre>']
(ROOT/'REPORT.html').write_text('\n'.join(parts),encoding='utf8')
print(json.dumps({'selected':report['selected']['selected'],'metrics':{p:v['aggregate'] for p,v in report['phases'].items()},'usage':report['usage']}))
