"""Exact-text BGE cache with transactional persistence of newly encoded vectors."""

if __name__ == '__main__':
    raise SystemExit('Historical dependency only. Use v8_run.py in the prepared DeepSeek benchmark.')

import hashlib
import json
from pathlib import Path
import re
import sqlite3
from threading import Lock

import numpy as np

MODEL = 'BAAI/bge-m3'
REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
DIMENSIONS = 1024


class CacheMissError(ValueError):
    pass


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def validate_vectors(vectors, count):
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (count, DIMENSIONS) or not np.isfinite(matrix).all():
        raise ValueError('Invalid BGE vector shape or values')
    if count and not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
        raise ValueError('BGE vectors must be L2 normalized')
    return matrix


class EmbeddingCache:
    def __init__(self, seed_vectors, seed_manifest, database, encoder=None, fresh=False):
        self.meta = json.loads(Path(seed_manifest).read_text())
        if (self.meta.get('model') != MODEL or self.meta.get('revision') != REVISION
                or self.meta.get('dimensions') != DIMENSIONS
                or self.meta.get('pooling') != 'L2-normalized CLS last_hidden_state[:,0]'):
            raise ValueError('Unexpected BGE model, revision or dimensions')
        with Path(seed_vectors).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != self.meta['cache_sha256']:
            raise ValueError('Corrupt embedding cache')
        mapping = self.meta['text_sha256_to_row']
        if (any(not re.fullmatch(r'[a-f0-9]{64}', key) for key in mapping)
                or any(type(row) is not int for row in mapping.values())
                or set(mapping.values()) != set(range(len(mapping)))):
            raise ValueError('Invalid embedding row mapping')
        with np.load(seed_vectors, allow_pickle=False) as data:
            matrix = validate_vectors(data['vectors'], len(mapping))
        self.vectors = {} if fresh else {key: matrix[row].tolist() for key, row in mapping.items()}
        self.encoder = encoder
        self.lock = Lock()
        self.new_texts = 0
        self.reused_texts = 0
        Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(database, check_same_thread=False)
        try:
            with self.db:
                self.db.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, value TEXT NOT NULL)')
                self.db.execute('CREATE TABLE IF NOT EXISTS vectors (hash TEXT PRIMARY KEY, vector TEXT NOT NULL)')
                contract = {'model': MODEL, 'revision': REVISION, 'dimensions': DIMENSIONS,
                            'pooling': self.meta.get('pooling'), 'schema_version': 1}
                row = self.db.execute('SELECT value FROM metadata WHERE id = 1').fetchone()
                if row is None:
                    self.db.execute('INSERT INTO metadata VALUES (1, ?)', (json.dumps(contract, sort_keys=True),))
                elif json.loads(row[0]) != contract:
                    raise ValueError('Persistent cache belongs to a different embedding configuration')
                saved = self.db.execute('SELECT hash, vector FROM vectors').fetchall()
                if fresh and saved:
                    raise ValueError('--fresh needs an empty database; existing vectors were not deleted')
                for key, value in saved:
                    if not re.fullmatch(r'[a-f0-9]{64}', key):
                        raise ValueError('Invalid persistent text hash')
                    vector = validate_vectors([json.loads(value)], 1)[0].tolist()
                    if key in self.vectors and not np.allclose(self.vectors[key], vector, atol=1e-5):
                        raise ValueError('Persistent cache conflicts with seed vectors')
                    self.vectors[key] = vector
        except Exception:
            self.db.close()
            raise

    def embed(self, texts):
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
            raise ValueError('Invalid input')
        keys = [text_hash(text) for text in texts]
        with self.lock:
            missing = {key: text for key, text in zip(keys, texts) if key not in self.vectors}
            if missing and self.encoder is None:
                raise CacheMissError(f'{len(missing)} exact texts absent from cache; encoding is disabled')
            items = list(missing.items())
            for offset in range(0, len(items), 8):
                batch = items[offset:offset + 8]
                matrix = validate_vectors(self.encoder([text for _, text in batch]), len(batch))
                encoded = {key: vector.tolist() for (key, _), vector in zip(batch, matrix)}
                # Commit before returning results; completed batches survive a later failure.
                with self.db:
                    self.db.executemany('INSERT INTO vectors VALUES (?, ?)',
                                        [(key, json.dumps(vector)) for key, vector in encoded.items()])
                self.vectors.update(encoded)
                self.new_texts += len(encoded)
            self.reused_texts += sum(key not in missing for key in keys)
            return [self.vectors[key] for key in keys], len(missing)

    def stats(self):
        with self.lock:
            return {'cached_texts': len(self.vectors), 'new_texts': self.new_texts,
                    'reused_texts': self.reused_texts, 'encoding_enabled': self.encoder is not None}

    def close(self):
        self.db.close()
