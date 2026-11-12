"""Versioned label corrections; the original dataset stays byte-for-byte frozen."""
import copy
import hashlib
import json

from benchmark import ROOT, cases, corpus


def label_hash():
    return hashlib.sha256((ROOT / 'label_errata.json').read_bytes()).hexdigest()


def corrected_cases(split='all'):
    errata = json.loads((ROOT / 'label_errata.json').read_text())
    if hashlib.sha256((ROOT / 'cases.jsonl').read_bytes()).hexdigest() != errata['dataset_sha256']:
        raise ValueError('Errata belongs to another dataset')
    rows = copy.deepcopy(cases())
    by_id = {row['id']: row for row in rows}
    docs = corpus()
    seen = set()
    for correction in errata['corrections']:
        key = correction['id']
        if key in seen or key not in by_id:
            raise ValueError('Unknown or duplicate label correction')
        seen.add(key)
        row = by_id[key]
        if row['reference_answer'] != correction['old_reference_answer']:
            raise ValueError('Unexpected previous reference answer')
        if correction['evidence_quote'] not in docs[correction['evidence_path']]:
            raise ValueError('Label correction has no literal source evidence')
        row['reference_answer'] = correction['reference_answer']
    return [row for row in rows if split == 'all' or row['split'] == split]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with open(args.output, 'x') as stream:
        for row in corrected_cases():
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
