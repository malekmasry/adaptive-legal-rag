"""Download exact official weights; no credential or remote Python code involved."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
REVISION='5617a9f61b028005a4858fdac845db406aefb181'
FILES=['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json']

def main():
    target=ROOT.parent/'bge_model'/REVISION;target.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://huggingface.co/api/models/BAAI/bge-m3/tree/'+REVISION,timeout=30) as r:
        catalog={x['path']:x for x in json.load(r)}
    saved=[]
    for name in FILES:
        expected=catalog[name];path=target/name
        started=time.monotonic()
        if not path.exists() or path.stat().st_size!=expected['size']:
            url='https://huggingface.co/BAAI/bge-m3/resolve/'+REVISION+'/'+name
            part=path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(url,timeout=60) as response,part.open('wb') as stream:
                total=0;last=0
                while chunk:=response.read(1024*1024):
                    stream.write(chunk);total+=len(chunk)
                    if time.monotonic()-last>20:
                        print(json.dumps({'file':name,'downloaded_MB':round(total/1e6),'total_MB':round(expected['size']/1e6)}),flush=True)
                        last=time.monotonic()
            assert part.stat().st_size==expected['size'],name
            part.replace(path)
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        if 'lfs' in expected:assert digest==expected['lfs']['oid'],name
        saved.append({'file':name,'bytes':path.stat().st_size,'sha256':digest})
        print(json.dumps({'file':name,'verified':True,'seconds':round(time.monotonic()-started,2)}),flush=True)
    (ROOT/'bge_weights.json').write_text(json.dumps({'model':'BAAI/bge-m3','revision':REVISION,'files':saved},indent=2)+'\n')

if __name__=='__main__':main()
