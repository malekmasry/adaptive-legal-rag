"""Twenty intentionally difficult existing development cases; fresh v8/hybrid runs."""
import argparse
import json
from pathlib import Path
import shutil
from prepare import prepare,sha,write,ROOT

IDS=['S01c','S04c','S10c','S12c','S20c','S22c','S24c','S35c',
     'S02c','S05c','S07a','S07b','S09c','S14a','S14b','S16c','S17a','S17b','S23c','S28c']
RATIONALE={
 'S01c':'dialogue / indefinite term','S04c':'hypothetical operands and penalty calculation',
 'S10c':'three subquestions with colloquial entity','S12c':'historical approval threshold / negation',
 'S20c':'instruction inside knowledge base and false user premise',
 'S22c':'no answer in corpus','S24c':'bounded legal answer / missing cross-referenced grounds',
 'S35c':'exception to ordinary probation period','S02c':'notice and increase-frequency false premise',
 'S05c':'payment deadline versus payment completion event',
 'S07a':'amendment not applicable before boundary','S07b':'amendment applicable after boundary',
 'S09c':'similar counterparties with different warranties','S14a':'old edition day before boundary',
 'S14b':'new edition from boundary date','S16c':'exception for unchanged approved template',
 'S17a':'dated letter not effective yet','S17b':'letter effective from boundary date',
 'S23c':'genuinely ambiguous entity; clarification appropriate',
 'S28c':'two separate numeric legal scenarios across documents'}


def main(a):
    root=a.output.resolve()
    prepare(a.source.resolve(),a.consolidated.resolve(),root,IDS,('conditional35',))
    plan=json.loads((root/'freeze.json').read_text(encoding='utf8'))
    plan.update(version='psb-platform-proxy-stress20-20261115',systems=['conditional35','batch_deadline_v8','hybrid35v8'],
        scope='20 deliberately adversarial historically exposed development questions; no representative leaderboard claim',
        selection_rationale=RATIONALE,baseline_donor_outputs='conditional35 saved historical outputs; v8 and hybrid generated fresh',
        api_limits={'chat_calls':320,'rerank_calls':110,'conservative_chat_token_ceiling':6000000})
    packages=root/'packages';packages.mkdir()
    expected={'batch_deadline_v8':('PSB_RAG_SUBMISSION_v8.zip','4088970c84806b37e2bcbbf4c6baa91943af4bee84222d061808d41cd3bc0d06'),
              'hybrid35v8':('PSB_RAG_RETRIEVER35_V8.zip','e81079b3eb98283e2339e8e2150dc6305ba5d5a714485ea6e0664baf2233a721')}
    plan['packages']={}
    for name,(file,digest) in expected.items():
        source=a.releases/file
        if sha(source)!=digest:raise ValueError('Runtime archive mismatch: '+name)
        shutil.copyfile(source,packages/(name+'.zip'));plan['packages'][name]={'sha256':digest,'top_k':40 if name=='batch_deadline_v8' else 12}
    shutil.copyfile(a.source/'knowledge-base.zip',root/'knowledge-base.zip')
    plan['input_hashes']['knowledge-base.zip']=sha(root/'knowledge-base.zip')
    for file in packages.iterdir():plan['input_hashes'][str(file.relative_to(root))]=sha(file)
    # Reuse completed parent controls/baselines only under the identical role
    # configuration, prompts, original question, gold and retrieval candidates.
    if a.parent:
        old=json.loads((a.parent/'freeze.json').read_text(encoding='utf8'))
        for name in ('judge.py','services.py'):
            if old['code_hashes'][name]!=sha(ROOT/name):raise ValueError('Cannot reuse under changed prompt/transport')
        old_cases={c['id']:c for c in json.loads((a.parent/'cases.json').read_text(encoding='utf8'))}
        new_cases={c['id']:c for c in json.loads((root/'cases.json').read_text(encoding='utf8'))}
        for identifier,case in old_cases.items():
            if new_cases.get(identifier)!=case:raise ValueError('Changed shared case')
        (root/'cache').mkdir();reused=[]
        for file in (a.parent/'cache').glob('*.json'):
            if file.name.startswith(('control-','baseline-','rerank-','conditional35-')):
                value=json.loads(file.read_text(encoding='utf8'))
                if value.get('status')=='ok':
                    shutil.copyfile(file,root/'cache'/file.name);reused.append(file.name)
        for name in ('calls.jsonl','service-identity.json'):
            shutil.copyfile(a.parent/name,root/name)
        plan['reuse']={'parent_freeze_sha256':sha(a.parent/'freeze.json'),'completed_cache_files':reused,
                       'spend_ledger_includes_parent':True,'old_judgments_for_new_generations_reused':False}
    engine=ROOT.parent/'v8_deepseek/engine'
    plan['embedding_code_hashes']={p:sha(engine/p) for p in ('embedding_cache.py','live_embedding_server.py','bge_weights.json')}
    write(root/'freeze.json',plan)
    print(json.dumps({'stress_cases':len(IDS),'systems':plan['systems'],'paid_calls_during_preparation':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','consolidated','output','releases'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent',type=Path);main(p.parse_args())
