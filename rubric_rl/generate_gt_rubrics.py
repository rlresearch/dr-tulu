#!/usr/bin/env python3
"""
Convert expert rubrics from HealthBench, DRB, and ResearchQA into a unified
weighted-criteria JSONL format for scoring by any judge model.

Usage:
    python generate_gt_rubrics.py \
        --benchmark healthbench \
        --rollouts /path/to/rollouts.jsonl \
        --output /path/to/gt_rubrics.jsonl

    python generate_gt_rubrics.py \
        --benchmark drb \
        --rollouts /path/to/rollouts.jsonl \
        --criteria /path/to/criteria.jsonl \
        --output /path/to/gt_rubrics.jsonl

    python generate_gt_rubrics.py \
        --benchmark researchqa \
        --rollouts /path/to/rollouts.jsonl \
        --rqa_data /path/to/test.json \
        --output /path/to/gt_rubrics.jsonl
"""
import argparse
import json
from pathlib import Path


def convert_healthbench(rollouts_path, output_path):
    with open(rollouts_path) as f:
        content = f.read().strip()
        data = json.loads(content) if content.startswith("[") else [json.loads(l) for l in content.split("\n") if l.strip()]

    count = 0
    with open(output_path, "w") as f:
        for ex in data:
            pid = ex["original_data"]["prompt_id"]
            rubrics = ex["original_data"]["rubrics"]
            pos_total = sum(r["points"] for r in rubrics if r["points"] > 0)
            if pos_total == 0:
                continue
            criteria = []
            for r in rubrics:
                weight = round(r["points"] / pos_total, 3)
                if r["points"] > 0:
                    criteria.append({"criterion": r["criterion"], "weight": weight,
                                     "scoring_levels": {"1.0": "Fully met", "0.0": "Not met"}})
                else:
                    criteria.append({"criterion": f"[DEMERIT] {r['criterion']}", "weight": weight,
                                     "scoring_levels": {"1.0": "This flaw is absent", "0.0": "This flaw is present"}})
            f.write(json.dumps({"prompt_id": pid, "question": ex["problem"],
                                "generated_rubric": json.dumps({"criteria": criteria}),
                                "rubric_style": "gt_expert", "rubric_model": "expert"}) + "\n")
            count += 1
    print(f"HealthBench: {count} GT rubrics -> {output_path}")


def convert_drb(rollouts_path, criteria_path, output_path):
    with open(rollouts_path) as f:
        rollouts = [json.loads(l) for l in f if l.strip()]
    with open(criteria_path) as f:
        crits = [json.loads(l) for l in f if l.strip()]
    crit_map = {c["prompt"]: c for c in crits}

    count = 0
    with open(output_path, "w") as f:
        for ex in rollouts:
            eid = str(ex["example_id"])
            crit_item = crit_map.get(ex["problem"])
            if not crit_item:
                continue
            dim_weights = crit_item["dimension_weight"]
            criterions = crit_item["criterions"]
            criteria = []
            for dim, dim_crits in criterions.items():
                dim_w = dim_weights.get(dim, 0.25)
                n_dim = len(dim_crits) if dim_crits else 1
                per_crit_w = round(dim_w / n_dim, 3) if dim_crits else dim_w
                for c in (dim_crits or []):
                    crit_text = c.get("criterion", c.get("description", str(c)))
                    criteria.append({"criterion": f"[{dim}] {crit_text}", "weight": per_crit_w,
                                     "scoring_levels": {"1.0": "Fully met", "0.5": "Partially met", "0.0": "Not met"}})
            f.write(json.dumps({"prompt_id": eid, "question": ex["problem"],
                                "generated_rubric": json.dumps({"criteria": criteria}),
                                "rubric_style": "gt_expert", "rubric_model": "expert"}) + "\n")
            count += 1
    print(f"DRB: {count} GT rubrics -> {output_path}")


def convert_researchqa(rollouts_path, rqa_data_path, output_path):
    with open(rollouts_path) as f:
        rollouts = [json.loads(l) for l in f if l.strip()]
    with open(rqa_data_path) as f:
        rqa_data = json.load(f)
    rqa_map = {item["id"]: item for item in rqa_data}

    count = 0
    with open(output_path, "w") as f:
        for ex in rollouts:
            eid = ex["example_id"]
            orig_id = ex["original_data"].get("orig_id", eid)
            rqa_item = rqa_map.get(orig_id)
            if not rqa_item or "rubric" not in rqa_item:
                continue
            rubric_items = rqa_item["rubric"]
            n = len(rubric_items)
            if n == 0:
                continue
            weight = round(1.0 / n, 3)
            criteria = [{"criterion": r["rubric_item"], "weight": weight,
                         "scoring_levels": {"1.0": "Completely covered", "0.5": "Partially covered", "0.0": "Not covered"}}
                        for r in rubric_items]
            f.write(json.dumps({"prompt_id": str(eid), "question": ex["problem"],
                                "generated_rubric": json.dumps({"criteria": criteria}),
                                "rubric_style": "gt_expert", "rubric_model": "expert"}) + "\n")
            count += 1
    print(f"ResearchQA: {count} GT rubrics -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["healthbench", "drb", "researchqa"])
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--criteria", default=None, help="DRB criteria.jsonl path")
    parser.add_argument("--rqa_data", default=None, help="ResearchQA test.json path")
    args = parser.parse_args()

    if args.benchmark == "healthbench":
        convert_healthbench(args.rollouts, args.output)
    elif args.benchmark == "drb":
        if not args.criteria:
            raise ValueError("--criteria required for DRB")
        convert_drb(args.rollouts, args.criteria, args.output)
    elif args.benchmark == "researchqa":
        if not args.rqa_data:
            raise ValueError("--rqa_data required for ResearchQA")
        convert_researchqa(args.rollouts, args.rqa_data, args.output)


if __name__ == "__main__":
    main()
