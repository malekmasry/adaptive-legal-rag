from pathlib import Path
import yaml

PROMPTS=yaml.safe_load(Path(__file__).with_name('prompts.yaml').read_text(encoding='utf8'))
REQUIRED={'system','planner','generator','verifier','source_classifier','reranker','memory_summary'}
if not isinstance(PROMPTS,dict) or not REQUIRED<=set(PROMPTS) or any(not isinstance(PROMPTS[k],str) or not PROMPTS[k].strip() for k in REQUIRED):
    raise ValueError('Missing or invalid prompt configuration')
