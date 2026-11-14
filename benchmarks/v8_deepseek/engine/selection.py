"""Conservative experiment selection; this never changes the published score."""

if __name__ == '__main__':
    raise SystemExit('Historical dependency only. Use v8_run.py in the prepared DeepSeek benchmark.')

from collections import defaultdict
from benchmark import aggregate_scores, bootstrap_interval


def selection_summary(rows, scope='pilot'):
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate result IDs')
    for r in rows:
        if r.get('status')=='ok' and (type(r.get('score')) not in (int,float) or r['score'] not in [i/8 for i in range(9)]):
            raise ValueError('Invalid score')
    result=aggregate_scores(rows)
    result['scope']=scope
    result['family_bootstrap_95']=bootstrap_interval(rows)
    low=[r if r.get('status')=='ok' else {**r,'status':'ok','score':0} for r in rows]
    high=[r if r.get('status')=='ok' else {**r,'status':'ok','score':1} for r in rows]
    result['unknown_score_bounds']={'low':aggregate_scores(low)['score'],'high':aggregate_scores(high)['score']}
    low_ci=bootstrap_interval(low);high_ci=bootstrap_interval(high)
    result['conservative_family_bootstrap_95']={'low':low_ci['low'],'high':high_ci['high']} if low_ci and high_ci else None
    result['unknown_count']=sum(r.get('status')!='ok' for r in rows)
    # Unknown judgments remain excluded from MR as in the public rules. These
    # bounds separately show whether excluding them could reverse the choice.
    result['supported_direction']=None
    interval=result['conservative_family_bootstrap_95']; bounds=result['unknown_score_bounds']
    if scope in ('full_development','heldout') and result['coverage']>=.95 and interval:
        if interval['low']>.5 and bounds['low']>.5:result['supported_direction']='candidate'
        if interval['high']<.5 and bounds['high']<.5:result['supported_direction']='baseline'
    for key in ('family_id','category','question_type'):
        groups=defaultdict(list)
        for r in rows:groups[str(r.get(key,'unspecified'))].append(r)
        result['by_'+key]={k:aggregate_scores(v) for k,v in sorted(groups.items())}
    result['not_psb_prediction']=True
    return result
