#!/usr/bin/env python3
"""Convert GT rubrics from all 3 benchmarks to the same weighted-criteria JSONL format."""
import json
import sys
from pathlib import Path


def convert_healthbench(rollouts_path, output_path):
    with open(rollouts_path) as f:
        content = f.read().strip()
        data = json.loads(content) if content.startswith("[") else [json.loads(l) for l in content.split("\n") if l.strip()]

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
                    criteria.append({
                        "criterion": r["criterion"],
                        "weight": weight,
                        "scoring_levels": {"1.0": "Fully met", "0.0": "Not met"}
                    })
                else:
                    criteria.append({
                        "criterion": f"[DEMERIT] {r['criterion']}",
                        "weight": weight,
                        "scoring_levels": {"1.0": "This flaw is absent", "0.0": "This flaw is present"}
                    })
            f.write(json.dumps({
                "prompt_id": pid,
                "question": ex["problem"],
                "generated_rubric": json.dumps({"criteria": criteria}),
                "rubric_style": "gt_expert",
                "rubric_model": "expert",
            }) + "\n")
    print(f"HealthBench: {len(data)} GT rubrics -> {output_path}")


def convert_researchqa(rollouts_path, rqa_data_path, output_path):
    with open(rollouts_path) as f:
        rollouts = [json.loads(l) for l in f if l.strip()]

    with open(rqa_data_path) as f:
        rqa_data = json.load(f)
    rqa_map = {item["id"]: item for item in rqa_data}

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
            criteria = []
            for r in rubric_items:
                criteria.append({
                    "criterion": r["rubric_item"],
                    "weight": weight,
                    "scoring_levels": {"1.0": "Completely covered", "0.5": "Partially covered", "0.0": "Not covered"}
                })
            f.write(json.dumps({
                "prompt_id": str(eid),
                "question": ex["problem"],
                "generated_rubric": json.dumps({"criteria": criteria}),
                "rubric_style": "gt_expert",
                "rubric_model": "expert",
            }) + "\n")
    print(f"ResearchQA: {len(rollouts)} -> {output_path}")


def convert_drb(rollouts_path, criteria_path, output_path):
    with open(rollouts_path) as f:
        rollouts = [json.loads(l) for l in f if l.strip()]

    with open(criteria_path) as f:
        crits = [json.loads(l) for l in f if l.strip()]
    crit_map = {}
    for c in crits:
        crit_map[c["prompt"]] = c

    with open(output_path, "w") as f:
        for ex in rollouts:
            eid = str(ex["example_id"])
            problem = ex["problem"]
            crit_item = crit_map.get(problem)
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
                    criteria.append({
                        "criterion": f"[{dim}] {crit_text}",
                        "weight": per_crit_w,
                        "scoring_levels": {"1.0": "Fully met", "0.5": "Partially met", "0.0": "Not met"}
                    })

            f.write(json.dumps({
                "prompt_id": eid,
                "question": problem,
                "generated_rubric": json.dumps({"criteria": criteria}),
                "rubric_style": "gt_expert",
                "rubric_model": "expert",
            }) + "\n")
    print(f"DRB: {len(rollouts)} -> {output_path}")


if __name__ == "__main__":
    base = "/checkpoint/dream/rulin/dr-tulu"
    out = f"{base}/rubric_rl/outputs/v3_comparison"

    convert_healthbench(
        f"{base}/eval_output/dr_tulu_8b/healthbench.jsonl",
        f"{out}/healthbench_gt_expert.jsonl",
    )
    convert_researchqa(
        f"{out}/researchqa_rl_overlap.jsonl",
        f"{base}/agent/evaluation/research_qa_eval/data/test.json",
        f"{out}/researchqa_gt_expert.jsonl",
    )
    convert_drb(
        f"{base}/agent/eval_output/dr_tulu_8b_step4000/deep_research_bench.jsonl",
        f"{base}/agent/evaluation/deep_research_bench_eval/data/criteria_data/criteria.jsonl",
        f"{out}/drb_gt_expert.jsonl",
    )
