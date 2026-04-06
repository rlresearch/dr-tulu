#!/usr/bin/env python3
"""
Score RL + SFT rollouts with generated rubrics using the frozen judge,
then compute pairwise distinguishability. Works for any benchmark.

Usage:
    python score_pairwise.py \
        --rl_rollouts /path/to/rl_rollouts.jsonl \
        --sft_rollouts /path/to/sft_rollouts.jsonl \
        --rubrics /path/to/rubrics.jsonl \
        --output /path/to/results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import vllm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from rubric_chat_templates import format_messages


def _strip_thinking_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def parse_judge_score(text: str) -> float | None:
    cleaned = _strip_thinking_block(text)
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start != -1 and json_end != -1:
        json_str = cleaned[json_start:json_end + 1]
        try:
            import json_repair
            data = json_repair.loads(json_str)
            if isinstance(data, dict) and "score" in data:
                return float(data["score"])
        except Exception:
            pass
        try:
            data = json.loads(json_str)
            if "score" in data:
                return float(data["score"])
        except Exception:
            pass
    match = re.search(r'"score"\s*:\s*([\d.]+)', cleaned)
    if match:
        return float(match.group(1))
    return None


def load_rollouts(path: str) -> dict:
    with open(path) as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split("\n") if line.strip()]
    return {str(d.get("example_id", d.get("original_data", {}).get("prompt_id", ""))): d for d in data}


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
    return [parse_judge_score(o.outputs[0].text.strip()) for o in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_rollouts", required=True)
    parser.add_argument("--sft_rollouts", required=True)
    parser.add_argument("--rubrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge_model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--judge_temperature", type=float, default=0.6)
    parser.add_argument("--judge_max_tokens", type=int, default=16384)
    parser.add_argument("--judge_max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    args = parser.parse_args()

    rl_data = load_rollouts(args.rl_rollouts)
    sft_data = load_rollouts(args.sft_rollouts)
    with open(args.rubrics) as f:
        rubric_data = {r["prompt_id"]: r for r in (json.loads(l) for l in f)}

    overlap = sorted(set(rl_data) & set(sft_data) & set(rubric_data))
    print(f"Matched {len(overlap)} examples")

    rl_examples, sft_examples = [], []
    for eid in overlap:
        rubric = rubric_data[eid]["generated_rubric"]
        question = rl_data[eid]["problem"]
        rl_examples.append({"question": question, "answer": rl_data[eid]["final_response"], "rubric": rubric})
        sft_examples.append({"question": question, "answer": sft_data[eid]["final_response"], "rubric": rubric})

    print(f"Loading frozen judge: {args.judge_model}")
    engine = vllm.LLM(
        model=args.judge_model, tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.judge_max_model_len, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)

    print("Scoring RL rollouts...")
    rl_scores = score_batch(rl_examples, engine, tokenizer, args.judge_temperature, args.judge_max_tokens)
    print(f"  Valid: {sum(1 for s in rl_scores if s is not None)}/{len(rl_scores)}")

    print("Scoring SFT rollouts...")
    sft_scores = score_batch(sft_examples, engine, tokenizer, args.judge_temperature, args.judge_max_tokens)
    print(f"  Valid: {sum(1 for s in sft_scores if s is not None)}/{len(sft_scores)}")

    paired = [(eid, rs, ss) for eid, rs, ss in zip(overlap, rl_scores, sft_scores) if rs is not None and ss is not None]
    rl_arr = np.array([p[1] for p in paired])
    sft_arr = np.array([p[2] for p in paired])
    diffs = rl_arr - sft_arr

    rl_wins = int((diffs > 0).sum())
    sft_wins = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())
    non_ties = rl_wins + sft_wins

    print(f"\n{'='*60}")
    print(f"PAIRWISE RESULTS (n={len(paired)})")
    print(f"{'='*60}")
    print(f"RL:  mean={rl_arr.mean():.4f}, std={rl_arr.std():.4f}")
    print(f"SFT: mean={sft_arr.mean():.4f}, std={sft_arr.std():.4f}")
    print(f"RL wins: {rl_wins} ({100*rl_wins/len(paired):.1f}%)")
    print(f"SFT wins: {sft_wins} ({100*sft_wins/len(paired):.1f}%)")
    print(f"Ties: {ties} ({100*ties/len(paired):.1f}%)")

    summary = {
        "n": len(paired), "rl_mean": float(rl_arr.mean()), "sft_mean": float(sft_arr.mean()),
        "rl_std": float(rl_arr.std()), "sft_std": float(sft_arr.std()),
        "mean_diff": float(diffs.mean()),
        "rl_wins": rl_wins, "sft_wins": sft_wins, "ties": ties,
        "rubric_model": rubric_data[overlap[0]].get("rubric_model", "unknown"),
        "judge_model": args.judge_model,
    }

    if non_ties > 0:
        from scipy import stats
        acc = rl_wins / non_ties
        binom_p = stats.binomtest(rl_wins, non_ties, 0.5, alternative='greater').pvalue
        t_stat, t_p = stats.ttest_rel(rl_arr, sft_arr)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diffs[diffs != 0])
        cohens_d = float(diffs.mean() / diffs.std()) if diffs.std() > 0 else 0.0

        print(f"\nPairwise accuracy: {acc:.3f} ({rl_wins}/{non_ties})")
        print(f"Binomial p: {binom_p:.2e}")
        print(f"Paired t-test: t={t_stat:.3f}, p={t_p:.2e}")
        print(f"Wilcoxon: p={wilcoxon_p:.2e}")
        print(f"Cohen's d: {cohens_d:.3f}")

        summary.update({
            "pairwise_accuracy": float(acc), "binomial_p": float(binom_p),
            "t_stat": float(t_stat), "t_p": float(t_p),
            "wilcoxon_p": float(wilcoxon_p), "cohens_d": cohens_d,
        })

        margins = {}
        for thresh in [0.0, 0.05, 0.10, 0.15, 0.20]:
            conf = np.abs(diffs) > thresh
            if conf.sum() > 0:
                c_acc = float((diffs[conf] > 0).sum() / conf.sum())
                margins[str(thresh)] = {"pairs": int(conf.sum()), "coverage": float(conf.sum()/len(paired)), "accuracy": c_acc}
                print(f"  |diff|>{thresh:.2f}: {conf.sum()} pairs ({conf.sum()/len(paired)*100:.0f}%), acc={c_acc:.3f}")
        summary["margins"] = margins

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_example": [{"id": eid, "rl_score": float(rs), "sft_score": float(ss)} for eid, rs, ss in paired]}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
