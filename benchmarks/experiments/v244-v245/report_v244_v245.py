"""Analyze frozen comparisons; never modify product, judge, or saved answers."""
import hashlib,html,json,statistics
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W/'v244-v245-experiment';OLD=W/'v242-experiment';OUT=W.parent/'outputs'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
cases=read(ROOT/'cases.json');baseline=read(ROOT/'baseline.json');result={'platform_reference':{'variant':'v243-final','score':37.5,'seconds':484,'source':'user_report'},'variants':{}}
def review_call(captured):
    return next((c for c in captured['calls'] if 'previous_draft' in c['payload'] and 'format_error' not in c['payload']),None)
for variant in ('v244','v245'):
    summary=read(ROOT/(variant+'-summary.json'));answers=read(ROOT/(variant+'-answers.json'));coverage=read(ROOT/(variant+'-coverage.json'))
    judgments={r['id']:r for r in read(ROOT/(variant+'-judgments.json'))};details=[]
    for case,cover in zip(cases,coverage):
        ident=case['id'];assert ident==cover['id']
        cap=read(ROOT/'generated'/(variant+'_regression')/'captured'/(ident+'.json'))
        old=read(OLD/'generated'/('v242_development' if ident.startswith('D') else 'v243_holdout')/'captured'/(ident+'.json'))
        current_review=review_call(cap);old_review=review_call(old)
        trace=(cap.get('result') or {}).get('trace',{});retrieval=trace.get('retrieval',{})
        details.append({'id':ident,'changed_answer':answers[ident]!=baseline[ident],
            'same_pre_review_draft':bool(current_review and old_review and current_review['payload']['previous_draft']==old_review['payload']['previous_draft']),
            'same_review_evidence':bool(current_review and old_review and current_review['payload']['evidence']==old_review['payload']['evidence']),
            'status':cap['result']['status'],'review':trace.get('review'),'patch_review':trace.get('patch_review'),
            'proposed_patches':current_review.get('response',{}).get('patches') if current_review else None,
            'route':retrieval.get('route'),'route_reason':retrieval.get('route_reason'),'lexical_max_score':retrieval.get('lexical_max_score'),
            'rerank_passes':int(bool(retrieval.get('lexical_rerank')))+int(bool(retrieval.get('dense_rerank'))) if variant=='v244' else 1,
            'degraded':retrieval.get('degraded',[]),'all_anchors_present':cover['all_anchors_present'],
            'context_tokens':cover['evidence_tokens'],'score':judgments[ident].get('score'),
            'initial_answer':cap.get('initial_answer')})
    from collections import Counter
    result['variants'][variant]={'summary':summary,'answers_changed':sum(d['changed_answer'] for d in details),
        'same_pre_review_draft':sum(d['same_pre_review_draft'] for d in details),
        'same_review_evidence':sum(d['same_review_evidence'] for d in details),
        'paired_review_cases':[d['id'] for d in details if d['same_pre_review_draft'] and d['same_review_evidence']],
        'routes':dict(Counter(d['route_reason'] for d in details)),'review_statuses':dict(Counter(d['review'] for d in details)),
        'required_anchors':sum(len(c['anchors']) for c in coverage),
        'present_anchors':sum(a['present'] for c in coverage for a in c['anchors']),
        'mean_context_tokens':statistics.mean(c['evidence_tokens'] for c in coverage),
        'wins':[d['id'] for d in details if d['score'] is not None and d['score']>.5],
        'losses':[d['id'] for d in details if d['score'] is not None and d['score']<.5],
        'details':details}
ledger=[json.loads(s) for s in (ROOT/'calls.jsonl').read_text(encoding='utf8').splitlines()]
done=[r for r in ledger if r['event']=='completed'];usage={}
for role in ('GENERATOR','JUDGE','RERANKER'):
    rows=[r for r in done if r['role']==role]
    usage[role]={'calls':len(rows),'successful':sum(r.get('status')=='ok' for r in rows),
                'tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in rows),
                'rerank_input_tokens':sum(r.get('meta',{}).get('tokens',{}).get('input_tokens',0) for r in rows)}
result['usage']=usage;result['api_errors']=[{'role':r['role'],'status':r.get('http_status'),'type':r.get('error_type')} for r in done if r.get('status')!='ok']
result['manual_audit']=read(ROOT/'manual-audit.json') if (ROOT/'manual-audit.json').exists() else {'status':'pending'}
result['limitations']=['All 30 cases were previously exposed; this is regression evidence, not independent holdout.',
    '50 is a pairwise tie against v243, not accuracy or a predicted PSB score.',
    'Shared exact caches reduce spend and distort wall-time comparisons with cold platform execution.',
    'The judge may disagree between numeric score and written rationale; all raw rounds are preserved.',
    'Initial draft/evidence equality is measured per question; only matched cases isolate the reviewer effect.']
write(ROOT/'analysis.json',result);write(OUT/'v244-v245-results.json',result)
esc=lambda x:html.escape(str(x))
parts=['<!doctype html><meta charset="utf-8"><title>v244 / v245</title><style>body{font:16px/1.55 system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#182536}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #c9d3df;padding:8px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;background:#f3f6f9;padding:14px;font:inherit}summary{font-weight:600;cursor:pointer;margin:12px 0}.note{background:#fff2cf;padding:16px}</style>',
    '<h1>v244: условный поиск; v245: точечное ревью</h1>',
    '<p class="note">Обе версии сравниваются с замороженными ответами v243 на 30 уже известных вопросах. 50 = ничья; это не процент правильных ответов и не прогноз балла ПСБ. Рабочая платформенная точка отсчёта со слов пользователя: 37,5/100 и 484 с.</p>',
    '<p>v244 меняет только поисковую стратегию и отложенную подготовку векторов. v245 сохраняет гибридный поиск и заменяет переписывание ответа на точечные правки. Новых библиотек нет. После получения результатов исходники не менялись.</p>',
    '<table><tr><th>Версия</th><th>Баллы</th><th>Победы/поражения/ничьи</th><th>Время с кешами</th><th>Опорные фрагменты</th><th>Тот же черновик и источники</th></tr>']
for name,row in result['variants'].items():
    s=row['summary'];a=s['aggregate']
    parts.append(f'<tr><td>{name}</td><td>{a["score"]:.4f}</td><td>{a["wins"]}/{a["losses"]}/{a["ties"]}</td><td>{s["runtime"]["seconds"]} с</td><td>{row["present_anchors"]}/{row["required_anchors"]}</td><td>{len(row["paired_review_cases"])}/30</td></tr>')
parts.append('</table><p>Время измерено на трёх потоках с точными кешами запросов; сравнивать эти секунды с 484 с платформы как ускорение нельзя.</p>')
parts.append('<p class="note">Достоверный прирост качества не установлен. Единственная числовая победа v245 (D07) противоречит пояснению судьи первого раунда, которое предпочитает v243. Match Rate = 50,4167%; большой итоговый балл вызван Win Rate = 100% из одной победы без поражений. Сырые оценки сохранены без перезапуска и ручного пересчёта.</p>')
parts.append('<h2>Ручной разбор</h2><pre>'+esc(json.dumps(result['manual_audit'],ensure_ascii=False,indent=2))+'</pre>')
parts.append('<h2>Расход текущего этапа</h2><pre>'+esc(json.dumps(usage,ensure_ascii=False,indent=2))+'</pre>')
for c in cases:
    parts.append('<details><summary>'+esc(c['id']+' — '+c['question'])+'</summary><h3>Эталон</h3><pre>'+esc(c['golden'])+'</pre><h3>v243</h3><pre>'+esc(baseline[c['id']])+'</pre>')
    for variant in ('v244','v245'):
        answer=read(ROOT/(variant+'-answers.json'))[c['id']]
        judged=next(r for r in read(ROOT/(variant+'-judgments.json')) if r['id']==c['id'])
        detail=next(d for d in result['variants'][variant]['details'] if d['id']==c['id'])
        parts.append('<h3>'+variant+'</h3><pre>'+esc(answer)+'</pre><details><summary>Судья: оба порядка</summary><pre>'+esc(json.dumps(judged,ensure_ascii=False,indent=2))+'</pre></details><details><summary>Поиск / проверка / правки</summary><pre>'+esc(json.dumps(detail,ensure_ascii=False,indent=2))+'</pre></details>')
    parts.append('</details>')
text='\n'.join(parts)
(ROOT/'REPORT.html').write_text(text,encoding='utf8');(OUT/'v244-v245-REPORT.html').write_text(text,encoding='utf8')
print(json.dumps({'variants':{n:{'metric':r['summary']['aggregate'],'runtime':r['summary']['runtime']['seconds'],'routes':r['routes'],'reviews':r['review_statuses'],'paired':len(r['paired_review_cases']),'wins':r['wins'],'losses':r['losses']} for n,r in result['variants'].items()},'usage':usage},ensure_ascii=False))
