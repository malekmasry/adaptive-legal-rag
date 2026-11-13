"""Lossless source ingestion, structural chunks and explicitly evidenced relations."""
import hashlib, json, re, zipfile
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict, field
from security import sanitize

PARSER_VERSION='structural-v1.0'

def digest(s): return hashlib.sha256(s.encode('utf8')).hexdigest()
def norm(s): return re.sub(r'[‐‑–—−]', '-',s.lower().replace('ё','е'))

@dataclass
class Document:
    id: str
    filename: str
    title: str
    raw: str
    clean: str
    sha256: str
    metadata: dict
    removed: list

@dataclass
class Chunk:
    id: str
    doc_id: str
    locator: str
    start: int
    end: int
    text: str
    parent_id: str
    ordinal: int
    kind: str
    annotations: list
    prefix: str = ''
    instruction_spans: list = field(default_factory=list)
    @property
    def search_text(self): return self.prefix+'\n'+self.text

def metadata(name,raw):
    n=norm(name); law=('законы' in n or re.search(r'(?:НК|ТК|КоАП|152ФЗ|МРОТ)[_ ]',name))
    m={'domain':'law' if law else 'internal','valid_from':'','valid_to':'',
       'entity':'','inn':re.findall(r'ИНН\s*(\d{10,12})',raw),'type':'law' if law else 'document',
       'export_date':'2026-10-10' if law else '', 'original_filename':name}
    keys=[('доп соглашение','amendment','p17'),('договор поставки п-17','contract','p17'),
          ('договор поставки пт-05','contract','pt05'),('договор аренды','contract','a08'),
          ('регламент','regulation',''),('письмо','letter',''),('служебная','memo','')]
    for needle,typ,entity in keys:
        if needle in n: m.update(type=typ,entity=entity); break
    if m['type']=='regulation':
        m.update(valid_from='2023-01-10' if '2023' in n else '2025-03-01',
                 valid_to='2025-03-01' if '2023' in n else '')
    if m['type']=='letter': m['valid_from']='2025-04-21'
    if m['type']=='amendment': m.update(valid_from='2024-09-03',condition='specification_date > 2024-10-01',target_clause='3.2')
    if m['type']=='contract':
        m['valid_from']={'p17':'2024-02-12','pt05':'2023-06-20','a08':'2024-08-01'}[m['entity']]
        m['renewal_or_indefinite']=m['entity'] in ('a08','pt05')
    m['annotations']=[x.strip() for x in raw.splitlines() if ANNOTATION.search(x)]
    m['content_gaps']=[]
    if law:
        m['edition_scope']='Only the supplied export; absent historical/future wording cannot be reconstructed.'
        for note in m['annotations']:
            missing=re.search(r'дополнена?\s+пунктом\s+(\d+)',note,re.I)
            if missing and not re.search(r'(?m)^\s*'+re.escape(missing[1])+r'\.\s',raw):
                m['content_gaps'].append({'locator':'п. '+missing[1],'reason':'editorial notice exists but normative text is absent','notice':note})
    return m

ANNOTATION=re.compile(r'^\s*(?:ГАРАНТ|Информация об изменениях|См\. |См. текст|Федеральным законом|Постановлением|(?:Пункт|Часть|Статья|Абзац|Подпункт)\s+[^\n]{0,100}(?:изменен|изменён|дополнен)|С .*статья .*дополн)',re.I)
NUMBER=re.compile(r'^\s*(\d+(?:\.\d+)*)([.)])\s+')

def split_document(doc,max_chars=1800):
    lines=list(re.finditer(r'[^\n]*(?:\n|$)',doc.clean)); segments=[]
    start=0; locator='преамбула'; parent=''; top=''; annotations=[]
    def flush(end):
        nonlocal start
        if doc.clean[start:end].strip(): segments.append((start,end,locator,parent,list(annotations)))
        start=end
    for line in lines:
        text=line.group().strip()
        if not text: continue
        if ANNOTATION.search(text):
            annotations.append({'start':line.start(),'text':text})
        match=NUMBER.match(text)
        if match:
            number=match[1]; delim=match[2]
            # Nested numeric list items remain attached to their governing numbered point.
            if delim==')': continue
            flush(line.start())
            if doc.metadata['domain']=='internal':
                if '.' not in number and len(text)<140:
                    top='раздел '+number; locator=top; parent=''
                else: locator='п. '+number; parent=top
            else:
                locator='п. '+number; parent='преамбула'
            annotations=[]
    flush(len(doc.clean))
    result=[]; loc_ids={}
    for start,end,loc,parent,notes in segments:
        full_id=digest(doc.sha256+f':{start}:{end}:{PARSER_VERSION}')[:24]
        loc_ids.setdefault(loc,full_id)
        # Whole semantic parent retained even if long; retrieval uses bounded children.
        parts=[]
        cursor=start
        for line in re.finditer(r'[^\n]*(?:\n|$)',doc.clean[start:end]):
            stop=start+line.end()
            if stop-cursor>max_chars and start+line.start()>cursor:
                parts.append((cursor,start+line.start())); cursor=start+line.start()
        if cursor<end: parts.append((cursor,end))
        if len(parts)>1:
            result.append(Chunk(full_id,doc.id,loc,start,end,doc.clean[start:end],loc_ids.get(parent,''),len(result),'parent',notes))
        for a,b in parts:
            # Very long single paragraph is kept intact: do not sever offense from sanction.
            cid=full_id if len(parts)==1 else digest(doc.sha256+f':{a}:{b}:{PARSER_VERSION}')[:24]
            result.append(Chunk(cid,doc.id,loc,a,b,doc.clean[a:b],full_id if len(parts)>1 else loc_ids.get(parent,''),len(result),'passage',notes))
    for c in result:
        m=doc.metadata
        c.prefix=f'{doc.title} | {c.locator} | {m["domain"]} | {m["entity"]} | действует с {m["valid_from"]} до {m["valid_to"] or "не установлено"}'
        # Attach editorial notes by actual target number, not whichever block preceded them.
        for line in lines:
            if ANNOTATION.search(line.group()):
                target=re.search(r'(?:Пункт|Часть|Подпункт)\s+(\d+(?:\.\d+)*)',line.group(),re.I)
                if (target and c.locator=='п. '+target[1]) or (not target and c.start<=line.start()<c.end):
                    note={'start':line.start(),'text':line.group().strip()}
                    if note not in c.annotations:c.annotations.append(note)
    return result

def read_sources(data_dir):
    root=Path(data_dir)
    # The platform archive is authoritative. Loose notes or stale extracted
    # documents in DATA_DIR must not silently replace the supplied knowledge base.
    for name in ('knowledge-base.zip','knowledge_base.zip','kb.zip'):
        if (root/name).exists():
            with zipfile.ZipFile(root/name) as z:
                files=[i for i in z.infolist() if i.filename.lower().endswith('.txt')]
                if sum(i.file_size for i in files)>32*1024*1024: raise ValueError('KB too large')
                # Read in memory, never extract paths supplied by an archive.
                return [(i.filename,z.read(i).decode('utf-8-sig')) for i in sorted(files,key=lambda x:x.filename)]
    folders=[root/'knowledge-base',root/'knowledge_base',root]
    for folder in folders:
        if folder.is_dir():
            paths=sorted(folder.rglob('*.txt'))
            paths=[p for p in paths if 'requirements' not in p.name.lower()]
            if paths:
                return [(str(p.relative_to(folder)),p.read_bytes().decode('utf-8-sig')) for p in paths]
    raise FileNotFoundError('Knowledge base not found in DATA_DIR')

def ingest(data_dir):
    docs=[]; chunks=[]
    for filename,raw in read_sources(data_dir):
        filename=filename.replace('\\','/').rsplit('/',1)[-1]
        if any(d.filename==filename for d in docs):raise ValueError('Duplicate source filename')
        # Audit suspicious spans without removing anything from the connected corpus.
        # Protection belongs to the trust boundary and evidence validator, not a patched KB.
        _,removed=sanitize(raw); clean=raw; sha=digest(raw)
        removed=[{**span,'action':'flag_only_source_preserved'} for span in removed]
        title=next((x.strip() for x in raw.splitlines() if x.strip()),filename)
        d=Document(digest(filename)[:20],filename,title,raw,clean,sha,metadata(filename,raw),removed)
        docs.append(d); chunks.extend(split_document(d))
    relations=[]
    def link(src,dst,kind,condition,locator):
        if src and dst:
            evidence=next((c for c in chunks if c.doc_id==src.id and locator in c.locator),None)
            if not evidence: evidence=next(c for c in chunks if c.doc_id==src.id)
            relations.append({'source':src.id,'target':dst.id,'kind':kind,'condition':condition,'evidence':evidence.id})
    byname={d.filename:d for d in docs}
    config=yaml.safe_load(Path(__file__).with_name('relations.yaml').read_text(encoding='utf8'))
    for rule in config['relations']:
        if rule['kind'] not in {'amends','supplements','supersedes','cites'}:raise ValueError('Unknown relation type')
        link(byname.get(rule['source']),byname.get(rule['target']),rule['kind'],rule['condition'],rule['evidence_locator'])
    return docs,chunks,relations

def applicable(doc,as_of):
    m=doc.metadata
    return (not m['valid_from'] or m['valid_from']<=as_of) and (not m['valid_to'] or as_of<m['valid_to'])
