from prompt_config import PROMPTS
"""Sources and conversation text never acquire instruction authority."""
import re

PATTERNS = [
    r'внимание\s+для\s+ии', r'игнорируй(?:те)?\s+(?:все|предыдущ|остальн)',
    r'(?:ignore|disregard)\s+(?:all|previous|prior)\s+(?:instructions|documents)',
    r'(?:system|developer)\s*(?:prompt|message)\s*:',
    r'не\s+упоминай\s+(?:эту\s+)?инструкц',
    r'(?:assistant|ассистент)[,:]?\s+(?:отвечай|выведи|замени)',
]
DETECTOR = re.compile('|'.join(PATTERNS),re.I)

def sanitize(raw):
    chars = list(raw)
    spans=[]
    for m in re.finditer(r'.+(?:\n|$)',raw):
        if DETECTOR.search(m.group()):
            spans.append({'start':m.start(),'end':m.end(),'reason':'instruction_to_model'})
            for i in range(m.start(),m.end()):
                if chars[i] not in '\r\n': chars[i]=' '
    return ''.join(chars),spans

def suspicious(text):
    return bool(DETECTOR.search(text))

def redact(text):
    text=re.sub(r'(?i)(?:gw_sk_|sk-|bearer\s+)[a-z0-9_\-]{12,}','[REDACTED]',str(text))
    return re.sub(r'(?i)(api[_-]?key["\s:=]+)[^\s,}"]+',r'\1[REDACTED]',text)

TRUST=PROMPTS['system']
