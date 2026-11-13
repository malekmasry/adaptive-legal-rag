"""Build an allowlisted flat PSB ZIP. Never include secrets, local data or test stubs."""
import ast,hashlib,json,re,sys,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FILES=['solution.py','settings.py','schemas.py','security.py','corpus.py','storage.py',
       'model_api.py','retriever35.py','retriever35_core.py','retrieval.py','reasoning.py','pipeline.py','source_audit.py','token_budget.py','prompt_config.py','diagnostics.py','prompts.yaml','aliases.yaml','relations.yaml','requirements.txt']
PACKAGES={'pydantic':'pydantic','httpx':'httpx','numpy':'numpy','sqlalchemy':'SQLAlchemy',
          'rank_bm25':'rank-bm25','rapidfuzz':'rapidfuzz','sklearn':'scikit-learn','yaml':'PyYAML','tiktoken':'tiktoken'}

def build(destination=None):
    destination=Path(destination or ROOT.parent/'PSB_RAG_SUBMISSION.zip')
    allowed={x.strip() for x in (ROOT/'allowed_requirements.txt').read_text(encoding='utf-8-sig').splitlines() if x.strip() and not x.startswith('#')}
    required={x.strip() for x in (ROOT/'requirements.txt').read_text().splitlines() if x.strip()}
    if not required<=allowed:raise ValueError('Unapproved dependencies: '+str(required-allowed))
    names={x.split('==')[0].lower() for x in required}; modules={Path(n).stem for n in FILES if n.endswith('.py')}
    for filename in FILES:
        path=ROOT/filename;content=path.read_text(encoding='utf8')
        if re.search(r'(?:gw_sk_|sk-|nvapi-)[A-Za-z0-9_-]{24,}',content):raise ValueError('Credential-like literal: '+filename)
        if filename.endswith('.py'):
            tree=ast.parse(content)
            for node in ast.walk(tree):
                imported=[]
                if isinstance(node,ast.Import):imported=[a.name.split('.')[0] for a in node.names]
                if isinstance(node,ast.ImportFrom):imported=[(node.module or '').split('.')[0]]
                for module in imported:
                    if module not in sys.stdlib_module_names and module not in modules:
                        if PACKAGES.get(module,'').lower() not in names:raise ValueError('Unapproved import '+module)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in FILES:z.write(ROOT/name,name)
        from token_budget import encoder
        if encoder() is None:raise ValueError('Offline tokenizer vocabulary missing')
        z.write(ROOT/'cl100k_base.tiktoken','cl100k_base.tiktoken')
        # Root-level data files are supported by the organizer's webinar clarification.
        if (ROOT/'data'/'knowledge-base').exists():
            from settings import Settings
            from corpus import ingest
            from retriever35 import Retriever
            s=Settings(data_dir=ROOT/'data',cache_dir=ROOT/'cache')
            d,c,r=ingest(s.data_dir);retriever=Retriever(d,c,r,None,s)
            from source_audit import fingerprint,load_audit
            audit=ROOT/('source-audit-'+fingerprint(c)+'.json')
            if audit.exists() and load_audit(c,ROOT):z.write(audit,audit.name)
            vector_source=retriever.cache_path if retriever.cache_path.exists() else ROOT/retriever.cache_path.name
            if retriever.vectors is not None and vector_source.exists():
                z.write(vector_source,retriever.cache_path.name)
    if destination.stat().st_size>128*1024*1024:raise ValueError('Archive exceeds PSB limit')
    with zipfile.ZipFile(destination) as z:archive_files=z.namelist()
    report={'archive':destination.name,'bytes':destination.stat().st_size,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'files':archive_files,
            'dependencies_allowlisted':True,'flat':True,'secrets_excluded':True,
            'contains_corpus':False,'contains_test_substitutes':False}
    destination.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report));return destination

if __name__=='__main__':build()
