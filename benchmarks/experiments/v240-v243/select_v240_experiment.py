"""Apply the predeclared development-only selection rule exactly once."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'v240-experiment'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT/'selection.json').exists()
assert not any((ROOT/'generated'/(v+'_holdout')).exists() for v in ['v239','v240','v241'])
summary=read(ROOT/'development-summary.json');manual=read(ROOT/'manual-audit.json')
eligible=[];checks={}
for variant in ['v240','v241']:
    score=summary['systems'][variant+'_vs_v239'];runtime=summary['runtime'][variant+'_development']
    criteria={'runtime_clean':runtime['complete'] and not runtime['exit_code'] and not runtime['runtime_report'].get('error_counts'),
        'full_judge_coverage':score['evaluated']==score['attempted']==18,
        'no_confirmed_factual_regression':not manual['development'][variant]['confirmed_factual_regressions_vs_v239'],
        'score_above_fifty':score['score']>50}
    checks[variant]={'criteria':criteria,'metric':score}
    if all(criteria.values()):eligible.append((score['score'],variant=='v240',variant))
selected=max(eligible)[2] if eligible else 'v239'
shutil.copyfile(ROOT/'manual-audit.json',ROOT/'manual-development-audit.json')
result={'selected':selected,'created_utc':datetime.now(timezone.utc).isoformat(),
    'runtime_freeze_sha256':sha(ROOT/'runtime-freeze.json'),
    'development_summary_sha256':sha(ROOT/'development-summary.json'),
    'manual_development_audit_sha256':sha(ROOT/'manual-development-audit.json'),
    'checks':checks,'reason':'No deployable candidate exceeded the baseline on development; retain v239.' if not eligible else 'Highest eligible development score under frozen rule.',
    'holdout_started':False,'no_holdout_selection':True,
    'note':'Oracle is diagnostic and ineligible. All three deployable variants will still be measured on holdout without changing this decision.'}
(ROOT/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
