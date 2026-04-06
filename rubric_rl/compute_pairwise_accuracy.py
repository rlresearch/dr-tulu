#!/usr/bin/env python3
"""
Compute pairwise ranking accuracy between synthetic rubric scores and GT scores.

For each question, compares whether the synthetic rubric (scored by frozen judge)
and the GT rubric (scored by GPT-4.1 binary grading) agree on which response
(RL vs SFT) is better.

Reports Acc (all non-tie pairs) and Acc@k (pairs with GT score gap > k).

Usage:
    python compute_pairwise_accuracy.py \
        --gt_rl scores/hb_rl_gt.json \
        --gt_sft scores/hb_sft_gt.json \
        --synthetic scores/hb_synthetic.json \
        --delta 0.1

    # Compare multiple generators:
    python compute_pairwise_accuracy.py \
        --gt_rl scores/hb_rl_gt.json \
        --gt_sft scores/hb_sft_gt.json \
        --synthetic scores/hb_ours.json scores/hb_qwen.json scores/hb_gpt.json \
        --labels ours Qwen3-8B GPT-4.1 \
        --delta 0.1
"""
import argparse
import json


def load_gt_scores(path):
    """Load GT scores: {pid: float}."""
    with open(path) as f:
        return json.load(f)


def load_synthetic_scores(path):
    """Load synthetic pairwise scores: {pid: (rl_score, sft_score)}."""
    with open(path) as f:
        data = json.load(f)
    return {e["id"]: (e["rl_score"], e["sft_score"]) for e in data["per_example"]}


def compute_accuracy(gt_rl, gt_sft, syn_map, delta=0):
    """Compute pairwise ranking accuracy.

    Excludes:
    - GT ties (gt_rl == gt_sft)
    - Synthetic ties (syn_rl == syn_sft)
    - Pairs with |GT gap| <= delta

    Returns (accuracy%, n_pairs).
    """
    agree = total = 0
    for pid in set(gt_rl) & set(gt_sft) & set(syn_map):
        grl, gsft = gt_rl[pid], gt_sft[pid]
        srl, ssft = syn_map[pid]
        if grl is None or gsft is None or srl is None or ssft is None:
            continue
        gt_d = grl - gsft
        syn_d = srl - ssft
        if gt_d == 0 or syn_d == 0:
            continue
        if abs(gt_d) <= delta:
            continue
        total += 1
        if (gt_d > 0 and syn_d > 0) or (gt_d < 0 and syn_d < 0):
            agree += 1
    acc = agree / total * 100 if total else 0
    return acc, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_rl", required=True, help="GT scores for RL rollouts (JSON: {pid: float})")
    parser.add_argument("--gt_sft", required=True, help="GT scores for SFT rollouts (JSON: {pid: float})")
    parser.add_argument("--synthetic", nargs="+", required=True, help="Synthetic score files (JSON with per_example)")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each synthetic file")
    parser.add_argument("--delta", nargs="+", type=float, default=[0, 0.01, 0.05, 0.1],
                        help="GT gap thresholds for Acc@k (default: 0 0.01 0.05 0.1)")
    args = parser.parse_args()

    gt_rl = load_gt_scores(args.gt_rl)
    gt_sft = load_gt_scores(args.gt_sft)
    labels = args.labels or [f"model_{i}" for i in range(len(args.synthetic))]

    print(f"GT: {len(gt_rl)} RL, {len(gt_sft)} SFT scores")
    overlap_gt = set(gt_rl) & set(gt_sft)
    diffs = [gt_rl[p] - gt_sft[p] for p in overlap_gt if gt_rl[p] is not None and gt_sft[p] is not None]
    ties = sum(1 for d in diffs if d == 0)
    print(f"GT overlap: {len(overlap_gt)}, ties: {ties}")
    for d in args.delta:
        n = sum(1 for diff in diffs if diff != 0 and abs(diff) > d)
        print(f"  |gap| > {d}: {n} pairs")
    print()

    header = f"{'delta':>8s}"
    for label in labels:
        header += f"  {label:>14s}"
    print(header)
    print("-" * len(header))

    for d in args.delta:
        label_str = "Acc" if d == 0 else f"Acc@{d}"
        row = f"{label_str:>8s}"
        for syn_path in args.synthetic:
            syn_map = load_synthetic_scores(syn_path)
            acc, n = compute_accuracy(gt_rl, gt_sft, syn_map, d)
            row += f"  {acc:>5.1f}% (n={n:>3d})"
        print(row)


if __name__ == "__main__":
    main()
