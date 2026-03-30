#!/usr/bin/env python3
"""Upload rubric RL evaluation results to HuggingFace as a dataset."""
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "rl-rag/rubric_rl_results"
BASE = Path("/checkpoint/dream/rulin/dr-tulu/rubric_rl/outputs")

api = HfApi(token=os.environ.get("HF_TOKEN"))

create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

# --- Upload rubric files as splits ---
uploads = {
    # v0 rubrics
    "rubrics/v0_healthbench.jsonl": BASE / "healthbench_all_rubrics.jsonl",
    "rubrics/v0_drb.jsonl": BASE / "drb_rubrics.jsonl",
    # v3 comparison rubrics
    "rubrics/rubric_gen_8b_v3_healthbench_v3prompt.jsonl": BASE / "v3_comparison/healthbench_rubric_gen_8b_v3.jsonl",
    "rubrics/rubric_gen_8b_v3_drb_v3prompt.jsonl": BASE / "v3_comparison/drb_rubric_gen_8b_v3.jsonl",
    "rubrics/rubric_gen_8b_v3_healthbench_standard.jsonl": BASE / "v3_comparison/healthbench_rubric_gen_8b_v3_standard.jsonl",
    "rubrics/rubric_gen_8b_v3_drb_standard.jsonl": BASE / "v3_comparison/drb_rubric_gen_8b_v3_standard.jsonl",
    "rubrics/qwen3_8b_healthbench_v3prompt.jsonl": BASE / "v3_comparison/healthbench_qwen3_8b.jsonl",
    "rubrics/qwen3_8b_drb_v3prompt.jsonl": BASE / "v3_comparison/drb_qwen3_8b.jsonl",
    "rubrics/qwen3_8b_healthbench_standard.jsonl": BASE / "v3_comparison/healthbench_qwen3_8b_standard.jsonl",
    "rubrics/qwen3_8b_drb_standard.jsonl": BASE / "v3_comparison/drb_qwen3_8b_standard.jsonl",
    "rubrics/gpt41_healthbench_v3prompt.jsonl": BASE / "v3_comparison/healthbench_gpt41.jsonl",
    "rubrics/gpt41_drb_v3prompt.jsonl": BASE / "v3_comparison/drb_gpt41.jsonl",
    "rubrics/gpt41_healthbench_standard.jsonl": BASE / "v3_comparison/healthbench_gpt41_standard.jsonl",
    "rubrics/gpt41_drb_standard.jsonl": BASE / "v3_comparison/drb_gpt41_standard.jsonl",
    # v0 pairwise scores
    "scores/v0_drb_pairwise.json": BASE / "drb_pairwise_results.json",
    # v3 comparison scores
    "scores/rubric_gen_8b_v3_healthbench_v3prompt.json": BASE / "v3_comparison/scores_healthbench_rubric_gen_8b_v3.json",
    "scores/rubric_gen_8b_v3_drb_v3prompt.json": BASE / "v3_comparison/scores_drb_rubric_gen_8b_v3.json",
    "scores/rubric_gen_8b_v3_healthbench_standard.json": BASE / "v3_comparison/scores_healthbench_rubric_gen_8b_v3_standard.json",
    "scores/rubric_gen_8b_v3_drb_standard.json": BASE / "v3_comparison/scores_drb_rubric_gen_8b_v3_standard.json",
    "scores/qwen3_8b_healthbench_v3prompt.json": BASE / "v3_comparison/scores_healthbench_qwen3_8b.json",
    "scores/qwen3_8b_drb_v3prompt.json": BASE / "v3_comparison/scores_drb_qwen3_8b.json",
    "scores/qwen3_8b_healthbench_standard.json": BASE / "v3_comparison/scores_healthbench_qwen3_8b_standard.json",
    "scores/qwen3_8b_drb_standard.json": BASE / "v3_comparison/scores_drb_qwen3_8b_standard.json",
    "scores/gpt41_healthbench_v3prompt.json": BASE / "v3_comparison/scores_healthbench_gpt41.json",
    "scores/gpt41_drb_v3prompt.json": BASE / "v3_comparison/scores_drb_gpt41.json",
    "scores/gpt41_healthbench_standard.json": BASE / "v3_comparison/scores_healthbench_gpt41_standard.json",
    "scores/gpt41_drb_standard.json": BASE / "v3_comparison/scores_drb_gpt41_standard.json",
}

# Also upload correlation results
corr_base = Path("/checkpoint/dream/rulin/dr-tulu/eval_output/dr_tulu_8b")
for name in ["rubric_correlation_results.json", "rubric_correlation_results_unclipped.json", "rubric_correlation_results_final.json"]:
    p = corr_base / name
    if p.exists():
        uploads[f"scores/{name}"] = p

for dest, src in uploads.items():
    if src.exists():
        print(f"Uploading {src.name} -> {dest}")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
    else:
        print(f"SKIP (not found): {src}")

print(f"\nDone! https://huggingface.co/datasets/{REPO_ID}")
