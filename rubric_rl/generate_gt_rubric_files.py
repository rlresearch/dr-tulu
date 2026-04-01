#!/usr/bin/env python3
"""Convert HealthBench GT rubrics to the same JSONL format as synthetic rubrics,
so they can be scored by the same frozen judge via score_pairwise.py."""
import json
import sys
from pathlib import Path


def convert_gt_rubrics_to_text(rubrics: list[dict]) -> str:
    """Convert HealthBench binary rubrics to a weighted-criteria JSON string
    matching the format the frozen judge expects."""
    pos_total = sum(r["points"] for r in rubrics if r["points"] > 0)
    if pos_total == 0:
        return json.dumps({"criteria": []})

    criteria = []
    for r in rubrics:
        weight = round(r["points"] / pos_total, 3)
        criteria.append({
            "criterion": r["criterion"],
            "weight": weight,
            "scoring_levels": {
                "1.0": "Fully met" if r["points"] > 0 else "Flaw is absent (good)",
                "0.0": "Not met" if r["points"] > 0 else "Flaw is present (bad)",
            }
        })
    return json.dumps({"criteria": criteria})


def main():
    rollouts_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(rollouts_path) as f:
        content = f.read().strip()
        data = json.loads(content) if content.startswith("[") else [json.loads(l) for l in content.split("\n") if l.strip()]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in data:
            pid = ex["original_data"]["prompt_id"]
            rubrics = ex["original_data"]["rubrics"]
            rubric_text = convert_gt_rubrics_to_text(rubrics)
            record = {
                "prompt_id": pid,
                "question": ex["problem"],
                "generated_rubric": rubric_text,
                "rubric_style": "gt_healthbench",
                "rubric_model": "expert",
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(data)} GT rubrics to {output_path}")


if __name__ == "__main__":
    main()
