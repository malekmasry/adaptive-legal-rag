"""Local BGE-M3 endpoint: reuse exact vectors, encode only misses, persist them."""

if __name__ == '__main__':
    raise SystemExit('Historical dependency only. Use v8_run.py in the prepared DeepSeek benchmark.')

import argparse
import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time

from embedding_cache import EmbeddingCache, MODEL, REVISION


class LazyBGE:
    """Import/load the pinned CPU encoder only if an exact-text cache miss occurs."""
    def __init__(self, weights):
        self.weights = weights
        self.model = None

    def __call__(self, strings):
        if self.model is None:
            manifest = json.loads(Path(__file__).with_name('bge_weights.json').read_text())
            if manifest['model'] != MODEL or manifest['revision'] != REVISION:
                raise ValueError('Unexpected weight manifest')
            for item in manifest['files']:
                with (self.weights/item['file']).open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                        raise ValueError('BGE weight checksum mismatch: '+item['file'])
            import torch
            from transformers import AutoTokenizer, AutoModel
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.weights, local_files_only=True, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(self.weights, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.float32, attn_implementation='sdpa').eval()
        tokens = self.tokenizer(strings, padding=True, truncation=False, return_tensors='pt')
        if tokens['input_ids'].shape[1] > 8192:
            raise ValueError('Input exceeds BGE context')
        with self.torch.inference_mode():
            encoded = self.model(**tokens).last_hidden_state[:, 0]
            return self.torch.nn.functional.normalize(encoded, p=2, dim=1).cpu().numpy()


def make_handler(cache, log):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, response):
            body = json.dumps(response).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != '/health':
                self.reply(404, {'error': {'type': 'UnknownEndpoint'}})
                return
            self.reply(200, {'ready': True, 'model': MODEL, 'revision': REVISION,
                             'pid': os.getpid(), **cache.stats()})

        def do_POST(self):
            try:
                if self.path != '/v1/embeddings':
                    raise ValueError('Unknown endpoint')
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if data.get('model', '').lower() != MODEL.lower():
                    raise ValueError('Wrong model')
                texts = data['input']
                texts = [texts] if isinstance(texts, str) else texts
                start = time.perf_counter()
                vectors, missing = cache.embed(texts)
                log.write(json.dumps({'texts': len(texts), 'new_texts': missing,
                    'elapsed_ms': round((time.perf_counter()-start)*1000, 3),
                    'input_hashes': [hashlib.sha256(t.encode()).hexdigest() for t in texts]})+'\n')
                log.flush()
                self.reply(200, {'model': MODEL, 'data': [
                    {'index': i, 'embedding': vector} for i, vector in enumerate(vectors)]})
            except Exception as exc:
                self.reply(400, {'error': {'type': type(exc).__name__}})
    return Handler


def make_server(cache, log, port):
    return ThreadingHTTPServer(('127.0.0.1', port), make_handler(cache, log))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, help='Optional; without weights, cache misses fail explicitly')
    p.add_argument('--seed-vectors', type=Path, required=True)
    p.add_argument('--seed-manifest', type=Path, required=True)
    p.add_argument('--cache-db', type=Path, help='Persistent additions; defaults beside the log')
    p.add_argument('--port', type=int, default=8769)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--fresh', action='store_true', help='Explicit full re-encoding; requires a new cache database')
    args = p.parse_args()
    if args.fresh and args.weights is None:
        p.error('--fresh requires --weights')
    cache = EmbeddingCache(args.seed_vectors, args.seed_manifest,
        args.cache_db or args.log.parent/'embedding-cache.sqlite3',
        encoder=LazyBGE(args.weights) if args.weights else None, fresh=args.fresh)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('a', encoding='utf-8', buffering=1) as log:
        server = make_server(cache, log, args.port)
        print(json.dumps({'ready': True, 'model': MODEL, 'revision': REVISION,
                          **cache.stats(), 'port': server.server_port}), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
            cache.close()


if __name__ == '__main__':
    main()
