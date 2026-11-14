"""Assemble a NEW private benchmark directory. No tests, inference or calibration."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))
from model_roles import normalize, JUDGE_MODEL


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def prepare(source, services, output):
    manifest = json.loads((HERE / "source-manifest.json").read_text())
    if sha(source / "v8/freeze.json") != manifest["source_freeze_sha256"]:
        raise ValueError("Source freeze differs")
    if sha(source / "v8/plan.json") != manifest["source_plan_sha256"]:
        raise ValueError("Source plan differs")
    old_plan = json.loads((source / "v8/plan.json").read_text())
    config = normalize(json.loads(services.read_text()))
    if any(not isinstance(v, str) for v in config.values()):
        raise ValueError("Service settings must be strings")
    tree = ast.parse((HERE / "engine/run_matrix.py").read_text())
    env_node = next(node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "ENV_NAMES" for t in node.targets))
    # Frozen source declares ENV_NAMES = set("...".split()); read the literal only.
    allowed = set(ast.literal_eval(env_node.value.args[0].func.value).split())
    allowed |= {"AS_OF", "EMBEDDER_BASE_URL", "EMBEDDER_API_KEY"}
    allowed |= {role + suffix for role in ("GENERATOR", "JUDGE") for suffix in ("_MODEL", "_BASE_URL", "_API_KEY")}
    if set(config) - allowed or config.get("MAX_WORKERS") != "3":
        raise ValueError("Unknown settings or workers differ from fixed profile")
    files = dict(manifest["source_inputs"])
    files.update(manifest["source_implementations"])
    files.update({"v8/packages/" + name: digest for name, digest in manifest["packages"].items()})
    # Byte integrity checks are assembly safeguards, not benchmark/test execution.
    for name, digest in files.items():
        if sha(source / name) != digest:
            raise ValueError("Source artifact differs: " + name)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    root = output / "benchmark"
    root.mkdir()
    for name in files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    for p in (HERE / "engine").iterdir():
        if p.is_file():
            shutil.copyfile(p, root / p.name)
    shutil.copyfile(HERE / "services.example.json", root / "v8/services.example.json")
    write(output / "services.private.json", config)
    os.chmod(output / "services.private.json", 0o600)
    public = {k: v for k, v in config.items() if not k.endswith(("_KEY", "_BASE_URL"))}
    public.update(EMB_MODEL="BAAI/bge-m3", embedding_provider="local pinned CLS BGE-M3",
                  embedding_revision="5617a9f61b028005a4858fdac845db406aefb181")
    public["endpoint_fingerprints"] = {k: hashlib.sha256(v.encode()).hexdigest()
        for k, v in config.items() if k.endswith("_BASE_URL") and not k.startswith("EMB_")}
    plan = dict(old_plan)
    plan.update(judge_model=JUDGE_MODEL, public_settings=public,
                service_profile_sha256=hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest(),
                implementations={name: sha(root / name) for name in old_plan["implementations"]},
                generation_engine="v6 engine with explicit role validation and judge credential isolation",
                evaluation_version=manifest["version"], paid_calls_at_freeze=0,
                calibration_status="NOT_RUN", source_freeze_sha256=manifest["source_freeze_sha256"])
    write(root / "v8/plan.json", plan)
    readiness = {"version": manifest["version"], "status": "PREPARED_NOT_RUN",
                 "source_freeze_sha256": manifest["source_freeze_sha256"],
                 "packages": json.loads((root / "v8/baselines.json").read_text())["packages"],
                 "corpus_sha256": plan["corpus_sha256"], "implementations": plan["implementations"],
                 "generation_engine": plan["generation_engine"],
                 "services": {"public_settings": public, "service_profile_sha256": plan["service_profile_sha256"]}}
    write(root / "v8/baselines.json", readiness)
    paths = set(manifest["source_inputs"]) | {p.name for p in (HERE / "engine").iterdir() if p.is_file()}
    frozen = {"at": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(root / "v8/plan.json"),
              "inputs": {name: sha(root / name) for name in sorted(paths)},
              "policy": "Fresh DeepSeek evaluation; no old judgments or calibration reused; inputs and rubric unchanged.",
              "assembly_only": True, "tests_run": False, "model_api_calls": 0}
    write(root / "v8/freeze.json", frozen)
    write(output / "assembly.json", {"version": manifest["version"], "judge_model": JUDGE_MODEL,
          "status": "PREPARED_NOT_RUN", "tests_run": False, "model_api_calls": 0,
          "old_results_imported": False, "freeze_sha256": sha(root / "v8/freeze.json")})
    print("Prepared new DeepSeek benchmark; calibration NOT_RUN; tests/API calls: 0")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-benchmark", type=Path, required=True)
    p.add_argument("--services", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        prepare(args.source_benchmark.resolve(), args.services.resolve(), args.output.resolve())
    except Exception as exc:
        # Do not print private configuration or credential values in error logs.
        print("Assembly failed: " + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
