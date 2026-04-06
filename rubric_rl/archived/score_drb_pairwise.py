#!/usr/bin/env python3
"""Score DRB rollouts (RL + SFT) with generated rubrics using the frozen judge,
then compute pairwise distinguishability."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import vllm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from rubric_chat_templates import format_messages
from run_rubric_and_judge import parse_judge_score, _strip_thinking_block


def score_batch(examples, engine, tokenizer, temperature=0.6, max_tokens=16384):
    prompts = []
    for ex in examples:
        rubric_for_judge = _strip_thinking_block(ex["rubric"])
        messages = format_messages("judge", {
            "question": ex["question"],
            "rubric": rubric_for_judge,
            "answer": ex["answer"],
        }, tokenize=False)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    params = vllm.SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
    outputs = engine.generate(prompts, params)
    return [parse_judge_score(o.outputs[0].text.strip())["score"] for o in outputs]


def main():
    rl_path = "/checkpoint/dream/rulin/dr-tulu/agent/eval_output/dr_tulu_8b_step4000/deep_research_bench.jsonl"
    sft_path = "/checkpoint/dream/rulin/dr-tulu/agent/eval_output/dr_tulu_sft_8b/deep_research_bench.jsonl"
    rubric_path = "/checkpoint/dream/rulin/dr-tulu/rubric_rl/outputs/drb_rubrics.jsonl"
    output_path = "/checkpoint/dream/rulin/dr-tulu/rubric_rl/outputs/drb_pairwise_results.json"

    with open(rl_path) as f:
        rl_data = {str(json.loads(l)["example_id"]): json.loads(l) for l in f}
    with open(sft_path) as f:
        sft_data = {str(json.loads(l)["example_id"]): json.loads(l) for l in f}
    with open(rubric_path) as f:
        rubric_data = {r["prompt_id"]: r for r in (json.loads(l) for l in f)}

    overlap = sorted(set(rl_data) & set(sft_data) & set(rubric_data))
    print(f"Matched {len(overlap)} examples")

    rl_examples = []
    sft_examples = []
    for eid in overlap:
        rubric = rubric_data[eid]["generated_rubric"]
        question = rl_data[eid]["problem"]
        rl_examples.append({"question": question, "answer": rl_data[eid]["final_response"], "rubric": rubric})
        sft_examples.append({"question": question, "answer": sft_data[eid]["final_response"], "rubric": rubric})

    print("Loading frozen judge: Qwen/Qwen3-1.7B")
    engine = vllm.LLM(model="Qwen/Qwen3-1.7B", tensor_parallel_size=1, gpu_memory_utilization=0.90, max_model_len=32768, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

    print("Scoring RL rollouts...")
    rl_scores = score_batch(rl_examples, engine, tokenizer)
    print(f"  Valid: {sum(1 for s in rl_scores if s is not None)}/{len(rl_scores)}")

    print("Scoring SFT rollouts...")
    sft_scores = score_batch(sft_examples, engine, tokenizer)
    print(f"  Valid: {sum(1 for s in sft_scores if s is not None)}/{len(sft_scores)}")

    paired = [(eid, rl_s, sft_s) for eid, rl_s, sft_s in zip(overlap, rl_scores, sft_scores) if rl_s is not None and sft_s is not None]
    rl_arr = np.array([p[1] for p in paired])
    sft_arr = np.array([p[2] for p in paired])
    diffs = rl_arr - sft_arr

    print(f"\n{'='*60}")
    print(f"DRB PAIRWISE RESULTS (n={len(paired)})")
    print(f"{'='*60}")
    print(f"DR-Tulu-8B (RL):  mean={rl_arr.mean():.4f}, std={rl_arr.std():.4f}")
    print(f"DR-Tulu-SFT-8B:   mean={sft_arr.mean():.4f}, std={sft_arr.std():.4f}")
    print(f"Mean diff (RL-SFT): {diffs.mean():.4f}")

    rl_wins = (diffs > 0).sum()
    sft_wins = (diffs < 0).sum()
    ties = (diffs == 0).sum()
    non_ties = rl_wins + sft_wins

    print(f"\nRL wins:  {rl_wins} ({100*rl_wins/len(paired):.1f}%)")
    print(f"SFT wins: {sft_wins} ({100*sft_wins/len(paired):.1f}%)")
    print(f"Ties:     {ties} ({100*ties/len(paired):.1f}%)")

    if non_ties > 0:
        from scipy import stats
        acc = rl_wins / non_ties
        binom_p = stats.binomtest(rl_wins, non_ties, 0.5, alternative='greater').pvalue
        t_stat, t_p = stats.ttest_rel(rl_arr, sft_arr)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diffs[diffs != 0])

        print(f"\nPairwise accuracy (excl ties): {acc:.3f} ({rl_wins}/{non_ties})")
        print(f"Binomial test p: {binom_p:.2e}")
        print(f"Paired t-test: t={t_stat:.3f}, p={t_p:.2e}")
        print(f"Wilcoxon: stat={wilcoxon_stat:.1f}, p={wilcoxon_p:.2e}")
        print(f"Cohen's d: {diffs.mean()/diffs.std():.3f}")

        for thresh in [0.0, 0.05, 0.10, 0.15, 0.20]:
            conf = np.abs(diffs) > thresh
            if conf.sum() > 0:
                c_acc = (diffs[conf] > 0).sum() / conf.sum()
                print(f"  |diff|>{thresh:.2f}: {conf.sum()} pairs, acc={c_acc:.3f}")

    results = {
        "summary": {
            "n": len(paired),
            "rl_mean": float(rl_arr.mean()), "sft_mean": float(sft_arr.mean()),
            "rl_wins": int(rl_wins), "sft_wins": int(sft_wins), "ties": int(ties),
            "pairwise_accuracy": float(rl_wins/non_ties) if non_ties else None,
        },
        "per_example": [{"id": eid, "rl_score": float(rs), "sft_score": float(ss)} for eid, rs, ss in paired],
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
