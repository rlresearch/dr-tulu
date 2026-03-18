#!/usr/bin/env python3
"""
Evaluate DR-Tulu rollouts with both generated rubrics (frozen judge) and
ground-truth rubrics (HealthBench GPT grader), then compute correlation.

Generated-rubric scoring uses the same model, prompts, and parsing as
rubric_rl training (Qwen/Qwen3-1.7B frozen judge via vLLM, with the "judge"
template from rubric_chat_templates.py).

Ground-truth rubric scoring uses the standard HealthBench binary-per-criterion
pipeline with an OpenAI grader model.

Usage:
    # Full run on a GPU node (needs 1 GPU for frozen judge):
    python evaluate_rubric_correlation.py \
        --rollouts /path/to/healthbench.jsonl \
        --rubrics /path/to/healthbench_all_rubrics.jsonl

    # Quick test:
    python evaluate_rubric_correlation.py \
        --rollouts /path/to/healthbench.jsonl \
        --rubrics /path/to/healthbench_all_rubrics.jsonl \
        --max-examples 20

    # Skip original rubric scoring (only generated):
    python evaluate_rubric_correlation.py \
        --rollouts /path/to/healthbench.jsonl \
        --rubrics /path/to/healthbench_all_rubrics.jsonl \
        --skip-original
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# rubric_rl utilities are imported lazily inside score_with_generated_rubric_batch
# to avoid requiring vLLM/transformers when only running original rubric scoring
sys.path.insert(0, str(Path(__file__).parent))

# Import HealthBench eval pipeline for original rubric scoring
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "evaluation"))

from evaluation.samplers.sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)

GRADER_TEMPLATE = """You are an expert evaluator. Below is a conversation between a user and an AI assistant, followed by a rubric item. Determine if the rubric item's criterion is met based on the conversation.

IMPORTANT: Output JSON with two fields:
- "criteria_met": true or false
- "explanation": a brief explanation

Conversation:
<<conversation>>

Rubric item:
<<rubric_item>>

Output:"""


def score_with_generated_rubric_batch(
    examples: list[dict],
    engine,
    tokenizer,
    temperature: float = 0.6,
    max_tokens: int = 16384,
) -> list[float | None]:
    """Score a batch of examples with generated rubrics using the frozen judge."""
    from rubric_chat_templates import format_messages
    from run_rubric_and_judge import parse_judge_score, _strip_thinking_block

    prompts = []
    for ex in examples:
        rubric_for_judge = _strip_thinking_block(ex["generated_rubric"])
        messages = format_messages(
            "judge",
            {
                "question": ex["question"],
                "rubric": rubric_for_judge,
                "answer": ex["answer"],
            },
            tokenize=False,
        )
        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if (
            hasattr(tokenizer, "chat_template")
            and tokenizer.chat_template is not None
            and "enable_thinking" in tokenizer.chat_template
        ):
            pass  # thinking enabled by default, matching training
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        prompts.append(prompt)

    import vllm
    params = vllm.SamplingParams(
        temperature=temperature, top_p=0.95, max_tokens=max_tokens
    )
    outputs = engine.generate(prompts, params)

    scores = []
    for output in outputs:
        raw = output.outputs[0].text.strip()
        result = parse_judge_score(raw)
        scores.append(result["score"])
    return scores


def score_with_original_rubrics(
    prompt_messages: list,
    answer: str,
    rubrics: list[dict],
    grader_sampler,
) -> float | None:
    """Score using the HealthBench binary-per-criterion pipeline."""
    total_positive = sum(r["points"] for r in rubrics if r["points"] > 0)
    if total_positive == 0:
        return None

    convo_str = "\n\n".join(
        f"{m['role']}: {m['content']}"
        for m in prompt_messages + [{"role": "assistant", "content": answer}]
    )

    achieved = 0.0
    for rubric_item in rubrics:
        rubric_str = f"[{rubric_item['points']}] {rubric_item['criterion']}"
        grader_prompt = GRADER_TEMPLATE.replace(
            "<<conversation>>", convo_str
        ).replace("<<rubric_item>>", rubric_str)

        messages = [{"content": grader_prompt, "role": "user"}]
        for _ in range(3):
            response = grader_sampler(messages)
            text = response.response_text
            json_start = text.find("{")
            json_end = text.rfind("}")
            if json_start != -1 and json_end != -1:
                try:
                    result = json.loads(text[json_start : json_end + 1])
                    if "criteria_met" in result and isinstance(
                        result["criteria_met"], bool
                    ):
                        if result["criteria_met"]:
                            achieved += rubric_item["points"]
                        break
                except json.JSONDecodeError:
                    continue

    return achieved / total_positive


def main():
    parser = argparse.ArgumentParser(description="Evaluate rubric correlation")
    parser.add_argument(
        "--rollouts", required=True, help="Path to DR-Tulu healthbench rollouts"
    )
    parser.add_argument(
        "--rubrics", required=True, help="Path to generated rubrics JSONL"
    )
    parser.add_argument("--output", default=None, help="Path to save results JSON")

    # Frozen judge settings (matching rubric_rl training)
    parser.add_argument(
        "--judge-model",
        default="Qwen/Qwen3-1.7B",
        help="Frozen judge model (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument("--judge-temperature", type=float, default=0.6)
    parser.add_argument("--judge-max-tokens", type=int, default=16384)
    parser.add_argument("--judge-max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)

    # HealthBench grader settings
    parser.add_argument(
        "--grader-model",
        default="gpt-4.1-mini",
        help="Model for original-rubric grading",
    )
    parser.add_argument(
        "--max-grader-workers",
        type=int,
        default=20,
        help="Concurrent API calls for original rubric grading",
    )

    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-generated", action="store_true")
    parser.add_argument("--skip-original", action="store_true")
    args = parser.parse_args()

    # --- Load data ---
    print("Loading data...")
    with open(args.rollouts) as f:
        content = f.read().strip()
        if content.startswith("["):
            rollouts = json.loads(content)
        else:
            rollouts = [json.loads(line) for line in content.split("\n") if line.strip()]

    rubric_map = {}
    with open(args.rubrics) as f:
        for line in f:
            r = json.loads(line)
            rubric_map[r["prompt_id"]] = r

    if args.max_examples:
        rollouts = rollouts[: args.max_examples]

    matched = []
    for r in rollouts:
        pid = r["original_data"]["prompt_id"]
        if pid in rubric_map:
            matched.append((r, rubric_map[pid]))
    print(f"Matched {len(matched)} examples (rollouts with generated rubrics)")

    # --- Score with generated rubrics (frozen judge via vLLM) ---
    gen_scores = {}
    if not args.skip_generated:
        print(f"\n=== Generated rubric scoring (frozen judge: {args.judge_model}) ===")
        from run_rubric_and_judge import create_engine
        from transformers import AutoTokenizer

        engine = create_engine(
            args.judge_model,
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
            args.judge_max_model_len,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.judge_model, trust_remote_code=True
        )

        examples_for_judge = []
        pids_for_judge = []
        for rollout, rubric_data in matched:
            pid = rollout["original_data"]["prompt_id"]
            examples_for_judge.append(
                {
                    "question": rollout["problem"],
                    "answer": rollout["final_response"],
                    "generated_rubric": rubric_data["generated_rubric"],
                }
            )
            pids_for_judge.append(pid)

        print(f"Scoring {len(examples_for_judge)} examples in batches of {args.batch_size}...")
        all_scores = []
        for i in range(0, len(examples_for_judge), args.batch_size):
            batch = examples_for_judge[i : i + args.batch_size]
            batch_scores = score_with_generated_rubric_batch(
                batch,
                engine,
                tokenizer,
                temperature=args.judge_temperature,
                max_tokens=args.judge_max_tokens,
            )
            all_scores.extend(batch_scores)
            valid = sum(1 for s in batch_scores if s is not None)
            print(
                f"  Batch {i // args.batch_size + 1}: {valid}/{len(batch)} valid scores"
            )

        for pid, score in zip(pids_for_judge, all_scores):
            gen_scores[pid] = score

        # Free GPU memory
        del engine
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        valid_gen = sum(1 for s in gen_scores.values() if s is not None)
        print(f"Generated rubric scoring complete: {valid_gen}/{len(gen_scores)} valid")

    # --- Score with original rubrics (HealthBench pipeline via OpenAI API) ---
    orig_scores = {}
    if not args.skip_original:
        print(f"\n=== Original rubric scoring (grader: {args.grader_model}) ===")
        grader_sampler = ChatCompletionSampler(
            model=args.grader_model,
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=1000,
            temperature=0,
        )

        def _score_one(item):
            rollout, _ = item
            pid = rollout["original_data"]["prompt_id"]
            prompt_messages = rollout["original_data"].get(
                "prompt", [{"role": "user", "content": rollout["problem"]}]
            )
            original_rubrics = rollout["original_data"]["rubrics"]
            try:
                score = score_with_original_rubrics(
                    prompt_messages,
                    rollout["final_response"],
                    original_rubrics,
                    grader_sampler,
                )
            except Exception as e:
                print(f"  [{pid[:8]}] grading failed: {e}")
                score = None
            return pid, score

        print(f"Scoring {len(matched)} examples with {args.max_grader_workers} workers...")
        with ThreadPoolExecutor(max_workers=args.max_grader_workers) as executor:
            futures = [executor.submit(_score_one, item) for item in matched]
            done = 0
            for future in as_completed(futures):
                pid, score = future.result()
                orig_scores[pid] = score
                done += 1
                if done % 50 == 0:
                    print(f"  Completed {done}/{len(matched)}")

        valid_orig = sum(1 for s in orig_scores.values() if s is not None)
        print(f"Original rubric scoring complete: {valid_orig}/{len(orig_scores)} valid")

    # --- Compute correlation ---
    results = []
    for rollout, rubric_data in matched:
        pid = rollout["original_data"]["prompt_id"]
        results.append(
            {
                "prompt_id": pid,
                "generated_score": gen_scores.get(pid),
                "original_score": orig_scores.get(pid),
            }
        )

    paired = [
        (r["generated_score"], r["original_score"])
        for r in results
        if r["generated_score"] is not None and r["original_score"] is not None
    ]

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({len(paired)} examples with both scores)")
    print(f"{'=' * 60}")

    if not args.skip_generated:
        gen_valid = [r["generated_score"] for r in results if r["generated_score"] is not None]
        if gen_valid:
            arr = np.array(gen_valid)
            print(f"Generated rubric scores:  mean={arr.mean():.3f}, std={arr.std():.3f}, n={len(arr)}")

    if not args.skip_original:
        orig_valid = [r["original_score"] for r in results if r["original_score"] is not None]
        if orig_valid:
            arr = np.array(orig_valid)
            print(f"Original rubric scores:   mean={arr.mean():.3f}, std={arr.std():.3f}, n={len(arr)}")

    summary = {}
    if len(paired) >= 2:
        from scipy import stats

        gen_arr = np.array([p[0] for p in paired])
        orig_arr = np.array([p[1] for p in paired])

        pearson_r, pearson_p = stats.pearsonr(gen_arr, orig_arr)
        spearman_r, spearman_p = stats.spearmanr(gen_arr, orig_arr)
        kendall_tau, kendall_p = stats.kendalltau(gen_arr, orig_arr)

        print(f"\nCorrelation (n={len(paired)}):")
        print(f"  Pearson r  = {pearson_r:.4f}  (p={pearson_p:.2e})")
        print(f"  Spearman ρ = {spearman_r:.4f}  (p={spearman_p:.2e})")
        print(f"  Kendall τ  = {kendall_tau:.4f}  (p={kendall_p:.2e})")

        summary = {
            "num_paired": len(paired),
            "generated_scores": {"mean": float(gen_arr.mean()), "std": float(gen_arr.std())},
            "original_scores": {"mean": float(orig_arr.mean()), "std": float(orig_arr.std())},
            "pearson": {"r": float(pearson_r), "p": float(pearson_p)},
            "spearman": {"r": float(spearman_r), "p": float(spearman_p)},
            "kendall": {"tau": float(kendall_tau), "p": float(kendall_p)},
            "judge_model": args.judge_model,
            "grader_model": args.grader_model,
        }
    elif paired:
        print("Only 1 paired example — cannot compute correlation.")
        summary = {"num_paired": 1, "error": "insufficient data"}
    else:
        print("No paired scores available.")
        summary = {"num_paired": 0}

    output_data = {"summary": summary, "per_example": results}

    output_path = args.output or str(
        Path(args.rollouts).parent / "rubric_correlation_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
