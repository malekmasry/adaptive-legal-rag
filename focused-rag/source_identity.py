"""Recognize formatting-only changes; never normalize words, punctuation or digits.

The live text remains unchanged. Canonical positions are mapped back to literal
live substrings before any source is given to a model.
"""
import hashlib
import re
from functools import lru_cache


def canonical(text):
    return ' '.join(text.split())


@lru_cache(maxsize=128)
def fingerprints(text):
    return (hashlib.sha256(text.encode('utf8')).hexdigest(),
            hashlib.sha256(canonical(text).encode('utf8')).hexdigest())


def source_matches(text, entry):
    exact, normalized = fingerprints(text)
    return (exact == entry.get('sha256') or
            bool(entry.get('canonical_sha256') and normalized == entry['canonical_sha256']))


@lru_cache(maxsize=32)
def canonical_view(text):
    pieces=[]; starts=[]; ends=[]
    for m in re.finditer(r'\S+', text):
        if pieces:
            pieces.append(' '); starts.append(ends[-1]); ends.append(m.start())
        pieces.append(m[0])
        starts.extend(range(m.start(),m.end())); ends.extend(range(m.start()+1,m.end()+1))
    return ''.join(pieces), starts, ends


def live_span(text, anchor):
    if not anchor or not canonical(anchor):
        return None
    position=text.find(anchor)
    if position>=0:
        return position,position+len(anchor),anchor
    normalized,starts,ends=canonical_view(text)
    needle=canonical(anchor); position=normalized.find(needle)
    if position<0 or normalized.find(needle,position+1)>=0:
        return None
    a,b=starts[position],ends[position+len(needle)-1]
    return a,b,text[a:b]


def canonical_offsets(text, start, end):
    from bisect import bisect_left
    normalized,starts,ends=canonical_view(text)
    a=bisect_left(ends,start+1); b=bisect_left(starts,end)
    while a<b and normalized[a].isspace():a+=1
    while b>a and normalized[b-1].isspace():b-=1
    assert normalized[a:b]==canonical(text[start:end])
    return a,b


def live_scope(text, unit, exact=True):
    if exact:
        a,b=unit['start'],unit['end']
    else:
        a,b=unit.get('canonical_start',-1),unit.get('canonical_end',-1)
        normalized,starts,ends=canonical_view(text)
        if not 0<=a<b<=len(normalized):return None
        a,b=starts[a],ends[b-1]
    if not 0<=a<b<=len(text):return None
    return a,b,text[a:b]
