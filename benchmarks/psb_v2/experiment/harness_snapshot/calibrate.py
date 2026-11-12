"""Compare frozen versions on shared questions; expose coverage and uncertainty.

No question selection or category weighting is fitted to leaderboard numbers.
An optional affine mapping is diagnostic and assessed leave-one-version-out.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

from benchmark import ROOT, aggregate_scores, cases


def score(values):
    return 100 * aggregate_scores([{"status": "ok", "score": value} for value in values])["score"]


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    for _, group in itertools.groupby(order, key=values.__getitem__):
        indexes = list(group)
        rank = statistics.mean(order.index(index) + 1 for index in indexes)
        for index in indexes:
            result[index] = rank
    return result


def correlation(x, y):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return statistics.correlation(x, y)


def affine(x, y):
    if len(x) < 2 or len(set(x)) < 2:
        return None
    xm, ym = statistics.mean(x), statistics.mean(y)
    slope = sum((a-xm)*(b-ym) for a, b in zip(x, y)) / sum((a-xm)**2 for a in x)
    return {"intercept": ym - slope*xm, "slope": slope}


def analyze(case_rows, judgments, official, repetitions=1000):
    ids = {row["id"] for row in case_rows}
    versions = list(judgments)
    valid = {}
    coverage = {}
    for version, rows in judgments.items():
        seen = {}
        for row in rows:
            if row["id"] not in ids or row["id"] in seen:
                raise ValueError("Duplicate or out-of-split judgment ID")
            if row.get("status") == "ok" and (type(row.get("score")) not in (float, int) or row["score"] not in [i/8 for i in range(9)]):
                raise ValueError("Invalid two-round score")
            seen[row["id"]] = row
        valid[version] = {key: row["score"] for key, row in seen.items() if row.get("status") == "ok"}
        unknown = len(ids) - len(valid[version])
        known = list(valid[version].values())
        coverage[version] = {"scored": len(known), "expected": len(ids), "coverage": len(known)/len(ids),
                             "unknown": unknown, "all_case_score_bounds": [score(known + [0]*unknown), score(known + [1]*unknown)]}
    common = sorted(set.intersection(*(set(rows) for rows in valid.values())))
    base = {"coverage": coverage, "common_cases": len(common), "expected_cases": len(ids),
            "common_coverage": len(common)/len(ids), "score_scale": "local proxy 0..100, not PSB accuracy"}
    if not common:
        return {**base, "status": "no_common_valid_cases", "ranking": []}
    local = {version: score([valid[version][key] for key in common]) for version in versions}
    ranking = sorted(versions, key=lambda version: local[version], reverse=True)
    groups = defaultdict(list)
    for row in case_rows:
        if row["id"] in common:
            groups[row["family_id"]].append(row["id"])
    pairs = list(itertools.combinations(versions, 2))
    distribution = {pair: [] for pair in pairs}
    rng = random.Random(20261113)
    for _ in range(repetitions):
        keys = [key for family in rng.choices(list(groups), k=len(groups)) for key in groups[family]]
        boot = {version: score([valid[version][key] for key in keys]) for version in versions}
        for pair in pairs:
            distribution[pair].append(boot[pair[0]] - boot[pair[1]])
    comparisons = []
    for a, b in pairs:
        samples = sorted(distribution[(a, b)])
        target = official[a] - official[b]
        comparisons.append({"a": a, "b": b, "official_gap": round(target, 4),
                            "local_gap": round(local[a]-local[b], 4),
                            "expected_direction_reproduced": (local[a]-local[b])*target > 0 if target else None,
                            "gap_bootstrap_95": [samples[int(.025*repetitions)], samples[min(repetitions-1,int(.975*repetitions))]] if len(groups)>1 else None,
                            "bootstrap_expected_direction_fraction": sum(delta*target>0 for delta in samples)/repetitions if target and len(groups)>1 else None})
    x, y = [local[v] for v in versions], [official[v] for v in versions]
    categories = defaultdict(list)
    for row in case_rows:
        if row['id'] in common:
            categories[row.get('category','unknown')].append(row['id'])
    breakdown = {category:{'cases':len(keys),'local_scores':{
        version:score([valid[version][key] for key in keys]) for version in versions}}
        for category,keys in categories.items()}
    mapping = affine(x, y)
    loo = []
    if len(versions) >= 4:
        for i, version in enumerate(versions):
            fit = affine(x[:i]+x[i+1:], y[:i]+y[i+1:])
            if fit and fit["slope"] > 0:
                prediction = fit["intercept"] + fit["slope"]*x[i]
                loo.append({"held_out_version": version, "predicted_official": prediction,
                            "reported_official": y[i], "absolute_error": abs(prediction-y[i])})
    return {**base, "status": "diagnostic_only", "ranking": ranking, "local_scores": local,
            "spearman": correlation(ranks(x), ranks(y)), "pairs": comparisons,
            "categories": breakdown,
            "family_bootstrap_units": len(groups), "affine_diagnostic": mapping,
            "leave_one_version_out": loo,
            "leave_one_version_out_mae": statistics.mean(row["absolute_error"] for row in loo) if len(loo)==len(versions) else None,
            "historical_order_reproduced": len(versions)>=4 and base["common_coverage"]>=.95 and
                all(row["expected_direction_reproduced"] for row in comparisons if abs(row["official_gap"])>2),
            "prospective_predictive_validity": "not_established",
            "warning": "Four aggregate scores cannot identify the hidden test. Bootstrap describes this local suite only; historical matching can overfit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", nargs="+", required=True, help="version_id=path.jsonl")
    parser.add_argument("--split", choices=["development", "holdout", "official_demo"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads((ROOT / "anchors.json").read_text())["versions"]
    official = {row["id"]: row["official_score"] for row in registry}
    judgments, metadata = {}, {}
    expected_hash = hashlib.sha256((ROOT / "cases.jsonl").read_bytes()).hexdigest()
    for value in args.judgments:
        version, raw_path = value.split("=", 1)
        if version not in official or version in judgments:
            raise ValueError("Unknown or repeated version")
        path = Path(raw_path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        if meta["split"] != args.split or meta["dataset_sha256"] != expected_hash:
            raise ValueError("Mismatched split or dataset")
        if not meta.get("candidate_run") or not meta.get("baseline_run"):
            raise ValueError("Calibration requires actual solution.py run metadata")
        run = meta["candidate_run"]
        if run.get("version_id") != version or run.get("entry_point") != "solution.py":
            raise ValueError("Wrong candidate version/entry point")
        if not run.get("service_check_passed") or not meta["baseline_run"].get("service_check_passed"):
            raise ValueError("Missing vector/reranker execution; this is not the intended pipeline")
        judgments[version] = [json.loads(line) for line in path.read_text().splitlines()]
        metadata[version] = meta
    # Never compare different question subsets, judges, baselines or services as
    # if their score difference were an architectural effect.
    keys = ("judge_model", "prompt_sha256", "judge_implementation_sha256", "baseline_sha256", "corpus_sha256", "ids")
    for key in keys:
        if len({json.dumps(meta[key], sort_keys=True) for meta in metadata.values()}) != 1:
            raise ValueError("Incomparable judgment runs: " + key)
    profiles = {meta[role]["service_profile_sha256"] for meta in metadata.values() for role in ('candidate_run','baseline_run')}
    if len(profiles) != 1:
        raise ValueError("Candidate service/model settings differ")
    expected_ids = set(next(iter(metadata.values()))["ids"])
    selected = [row for row in cases(args.split) if row["id"] in expected_ids]
    if not selected or {row['id'] for row in selected}!=expected_ids:
        raise ValueError('Invalid or empty evaluation subset')
    report = analyze(selected, judgments, official)
    report["historical_configs_confirmed"] = all(meta["candidate_run"].get("historical_config_confirmed") for meta in metadata.values())
    report["split"] = args.split
    report["legacy_holdout_caveat"] = args.split == "holdout"
    report["full_split_evaluated"] = len(selected) == len(cases(args.split))
    report["pilot_only"] = not report["full_split_evaluated"]
    report["ordering_matches_on_evaluated_subset"] = report.get("historical_order_reproduced", False)
    report["historical_order_reproduced"] = bool(report["ordering_matches_on_evaluated_subset"] and
        report["full_split_evaluated"] and report["historical_configs_confirmed"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: report.get(key) for key in ("status", "ranking", "common_coverage", "spearman", "leave_one_version_out_mae", "historical_order_reproduced", "historical_configs_confirmed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
