"""Offline approximate token count. cl100k is NOT the DeepSeek tokenizer."""
import base64,hashlib,math
from functools import lru_cache
from pathlib import Path
import tiktoken

VOCAB_SHA256='223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7'
PATTERN=r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"

@lru_cache(maxsize=1)
def encoder():
    path=Path(__file__).with_name('cl100k_base.tiktoken')
    if not path.exists():return None
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=VOCAB_SHA256:raise ValueError('Tokenizer vocabulary checksum mismatch')
    ranks={base64.b64decode(token):int(rank) for token,rank in (line.split() for line in data.splitlines() if line)}
    return tiktoken.Encoding(name='local_cl100k_estimate',pat_str=PATTERN,mergeable_ranks=ranks,special_tokens={})

def estimate_tokens(text,margin=1.35):
    enc=encoder()
    if enc is None:return len(text.encode('utf8')) # Offline conservative fallback; never downloads.
    return math.ceil(len(enc.encode_ordinary(text))*margin)

def method():return 'cl100k_base approximation + 35% reserve' if encoder() else 'UTF-8 byte upper bound'
