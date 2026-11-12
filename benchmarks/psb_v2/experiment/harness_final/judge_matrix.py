"""Judge completed generation runs, with bounded API concurrency and resume support."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
VERSIONS = ('vector14', 'conditional35', 'coverage2257', 'review29')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--services', type=Path, required=True)
    parser.add_argument('--model', default='qwen3.8-max')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--parallel-versions', type=int, default=2)
    parser.add_argument('--split', default='development')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    active, completed, reported = {}, set(), {}
    started = time.monotonic()

    def ready(version):
        path = args.runs / version / 'run.meta.json'
        if not path.exists():
            return False
        meta = json.loads(path.read_text())
        if 'exit_code' not in meta:
            return False
        if meta['exit_code'] or not meta.get('service_check_passed'):
            raise RuntimeError('Generation failed: ' + version)
        return True

    try:
        while len(completed) < len(VERSIONS):
            if time.monotonic() - started > 7200:
                raise TimeoutError('Judge matrix exceeded two hours')
            baseline_ready = ready('vector14')
            for version in VERSIONS:
                if version in completed or version in active:
                    continue
                if len(active) >= max(1, min(args.parallel_versions, 2)):
                    break
                if not baseline_ready or not ready(version):
                    continue
                log = (args.output / (version + '.launcher.txt')).open('a')
                command = [sys.executable, str(ROOT / 'judge_batch.py'),
                           '--candidate', str(args.runs / version / 'submission.csv'),
                           '--baseline', str(args.runs / 'vector14' / 'submission.csv'),
                           '--services', str(args.services), '--model', args.model,
                           '--output', str(args.output / (version + '.jsonl')),
                           '--split', args.split, '--workers', str(args.workers)]
                active[version] = (subprocess.Popen(command, stdout=log, stderr=log), log)
                print(json.dumps({'stage': 'judge_start', 'version': version}), flush=True)
            for version, (process, log) in list(active.items()):
                path = args.output / (version + '.jsonl')
                count = len(path.read_text().splitlines()) if path.exists() else 0
                if reported.get(version) != count:
                    print(json.dumps({'stage': 'judge_progress', 'version': version, 'finished': count}), flush=True)
                    reported[version] = count
                if process.poll() is not None:
                    log.close()
                    if process.returncode:
                        raise RuntimeError('Judge failed: ' + version)
                    completed.add(version)
                    del active[version]
                    print(json.dumps({'stage': 'judge_complete', 'version': version}), flush=True)
            if len(completed) < len(VERSIONS):
                time.sleep(5)
    finally:
        for process, log in active.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            log.close()


if __name__ == '__main__':
    main()
