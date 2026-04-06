#!/usr/bin/env python3
"""
Standalone script to generate a rubric from a question and score an answer using
the frozen judge model.

Pipeline:
  1. Load the rubric generator model (default: stellalisy/rubric_generator_v0_0302) via vLLM
  2. Generate a rubric for the given question
  3. Load the frozen judge model (default: Qwen/Qwen3-1.7B) via vLLM
  4. Score the provided answer against the rubric

Usage:
  # Generate rubric + judge an answer (two models)
  python scripts/run_rubric_and_judge.py \
      --question "Explain the theory of relativity." \
      --answer "E=mc^2 describes mass-energy equivalence."

  # Supply your own rubric and only run the judge
  python scripts/run_rubric_and_judge.py \
      --question "What is the capital of France?" \
      --answer "Paris is the capital of France." \
      --rubric "Answer must name the correct capital. Score: 1.0" \
      --judge_model Qwen/Qwen3-1.7B

  # Use correctness-focused rubric generation
  python scripts/run_rubric_and_judge.py \
      --question "Solve x^2 - 4 = 0" \
      --answer "x = 2 or x = -2" \
      --rubric_style correctness

  # Judge mode: binary YES/NO instead of 0-1 score
  python scripts/run_rubric_and_judge.py \
      --question "What is 2+2?" \
      --answer "4" \
      --judge_mode binary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import vllm
from transformers import AutoTokenizer

from rubric_chat_templates import format_messages


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a rubric and/or judge an answer using vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--question", required=True, help="The question to generate a rubric for / judge against.")
    p.add_argument("--answer", default=None, help="Answer to evaluate. If omitted, only the rubric is generated.")
    p.add_argument("--rubric", default=None, help="Pre-supplied rubric text. Skips rubric generation when provided.")

    # Models
    p.add_argument("--rubric_model", default="stellalisy/rubric_generator_v0_0302",
                   help="HF model ID for rubric generation.")
    p.add_argument("--judge_model", default="Qwen/Qwen3-1.7B",
                   help="HF model ID for the frozen judge.")

    # Generation knobs
    p.add_argument("--rubric_style", choices=["standard", "correctness"], default="standard",
                   help="Rubric generation prompt style.")
    p.add_argument("--judge_mode", choices=["score", "binary"], default="score",
                   help="Judge output format: JSON score (0-1) or binary YES/NO.")
    p.add_argument("--rubric_temperature", type=float, default=0.6)
    p.add_argument("--rubric_max_tokens", type=int, default=16384)
    p.add_argument("--judge_temperature", type=float, default=0.6)
    p.add_argument("--judge_max_tokens", type=int, default=16384)

    # Thinking mode (Qwen3 models support <think>...</think> reasoning blocks).
    # Training uses inline vLLM which calls apply_chat_template without
    # enable_thinking, so Qwen3 defaults to thinking ON for both models.
    p.add_argument("--rubric_enable_thinking", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable thinking for rubric generator (default: True, matches training).")
    p.add_argument("--judge_enable_thinking", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable thinking for judge model (default: True, matches training).")

    # vLLM
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=None,
                   help="Max sequence length. Defaults to 16384 for rubric model, 32768 for judge.")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    return p


def create_engine(model: str, tp: int, gpu_mem: float, max_len: int | None, trust_remote_code: bool) -> vllm.LLM:
    kwargs = dict(
        model=model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=trust_remote_code,
    )
    if max_len is not None:
        kwargs["max_model_len"] = max_len
    print(f"Loading model: {model}  (tp={tp}, gpu_mem={gpu_mem})")
    return vllm.LLM(**kwargs)


def _supports_enable_thinking(tokenizer) -> bool:
    """Check if the tokenizer's chat template accepts `enable_thinking`."""
    return (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
        and "enable_thinking" in tokenizer.chat_template
    )


def _strip_thinking_block(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3 models may emit."""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return cleaned.strip()


def generate(
    engine: vllm.LLM, tokenizer, messages: list[dict],
    temperature: float, max_tokens: int, disable_thinking: bool = False,
) -> tuple[str, str]:
    """Returns (cleaned_text, raw_text)."""
    template_kwargs: dict = dict(tokenize=False, add_generation_prompt=True)
    if _supports_enable_thinking(tokenizer) and disable_thinking:
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    params = vllm.SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
    outputs = engine.generate([prompt], params)
    raw = outputs[0].outputs[0].text.strip()
    cleaned = _strip_thinking_block(raw)
    return cleaned, raw


def parse_judge_score(text: str) -> dict:
    """Extract score and reasoning from the judge's JSON response.

    Mirrors training's extract_json_from_response: strips thinking blocks first,
    then finds the outermost { ... } and attempts json_repair -> json.loads
    fallbacks.
    """
    cleaned = _strip_thinking_block(text)

    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start != -1 and json_end != -1 and json_end >= json_start:
        json_str = cleaned[json_start : json_end + 1]
        try:
            import json_repair
            data = json_repair.loads(json_str)
            if isinstance(data, dict) and "score" in data:
                return {"score": float(data["score"]), "reasoning": data.get("reasoning", ""), "raw": text}
        except Exception:
            pass
        try:
            data = json.loads(json_str)
            return {"score": float(data["score"]), "reasoning": data.get("reasoning", ""), "raw": text}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    match = re.search(r'"score"\s*:\s*([\d.]+)', cleaned)
    if match:
        return {"score": float(match.group(1)), "reasoning": cleaned, "raw": text}
    return {"score": None, "reasoning": cleaned, "raw": text}


def parse_binary_judgment(text: str) -> dict:
    """Extract YES/NO from binary judge output."""
    match = re.search(r"<EVALUATION>\s*(YES|NO)\s*</EVALUATION>", text, re.IGNORECASE)
    verdict = match.group(1).upper() if match else None
    return {"verdict": verdict, "raw": text}


def main():
    args = build_parser().parse_args()

    need_rubric_gen = args.rubric is None
    need_judge = args.answer is not None
    same_model = need_rubric_gen and need_judge and args.rubric_model == args.judge_model

    rubric_text = args.rubric

    # --- Step 1: Rubric generation ---
    if need_rubric_gen:
        rubric_max_len = args.max_model_len or 16384
        rubric_engine = create_engine(args.rubric_model, args.tensor_parallel_size,
                                      args.gpu_memory_utilization, rubric_max_len, args.trust_remote_code)
        rubric_tokenizer = AutoTokenizer.from_pretrained(args.rubric_model, trust_remote_code=True)

        template = "rubric_generation_correctness" if args.rubric_style == "correctness" else "rubric_generation"
        messages = format_messages(template, {"question": args.question}, tokenize=False)

        print("\n" + "=" * 60)
        print("RUBRIC GENERATION")
        print("=" * 60)
        print(f"Model : {args.rubric_model}")
        print(f"Style : {args.rubric_style}")
        print(f"Question: {args.question}\n")

        rubric_text, rubric_raw = generate(rubric_engine, rubric_tokenizer, messages,
                                             args.rubric_temperature, args.rubric_max_tokens,
                                             disable_thinking=not args.rubric_enable_thinking)

        print("--- Raw Model Output ---")
        print(rubric_raw)
        print("--- End Raw Output ---\n")
        print("--- Parsed Rubric (thinking stripped) ---")
        print(rubric_text)
        print("--- End Rubric ---\n")
    else:
        rubric_engine = None
        rubric_tokenizer = None

    if not need_judge:
        print("No --answer provided; stopping after rubric generation.")
        return

    # --- Step 2: Judging ---
    if same_model:
        judge_engine = rubric_engine
        judge_tokenizer = rubric_tokenizer
    else:
        # Free the rubric engine before loading the judge if they are different models
        if rubric_engine is not None:
            del rubric_engine
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        judge_max_len = args.max_model_len or 32768
        judge_engine = create_engine(args.judge_model, args.tensor_parallel_size,
                                     args.gpu_memory_utilization, judge_max_len, args.trust_remote_code)
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)

    # Strip any residual thinking tokens from the rubric before passing to judge,
    # matching training's _remove_redacted_reasoning() in rubric_judge_rewards.py.
    rubric_for_judge = _strip_thinking_block(rubric_text)

    if args.judge_mode == "binary":
        template = "judge_binary"
    else:
        template = "judge"

    messages = format_messages(template, {
        "question": args.question,
        "rubric": rubric_for_judge,
        "answer": args.answer,
    }, tokenize=False)

    print("=" * 60)
    print("JUDGE EVALUATION")
    print("=" * 60)
    print(f"Model : {args.judge_model}")
    print(f"Mode  : {args.judge_mode}")
    print(f"Answer: {args.answer[:200]}{'...' if len(args.answer) > 200 else ''}\n")

    judge_output, judge_raw = generate(judge_engine, judge_tokenizer, messages,
                                        args.judge_temperature, args.judge_max_tokens,
                                        disable_thinking=not args.judge_enable_thinking)

    print("--- Raw Judge Output ---")
    print(judge_raw)
    print("--- End Raw Output ---\n")
    print("--- Cleaned Judge Output (thinking stripped) ---")
    print(judge_output)
    print("--- End Cleaned Output ---\n")

    if args.judge_mode == "binary":
        result = parse_binary_judgment(judge_output)
        print(f"Verdict: {result['verdict']}")
    else:
        result = parse_judge_score(judge_output)
        if result["score"] is not None:
            print(f"Score : {result['score']:.2f}")
            print(f"Reason: {result['reasoning']}")
        else:
            print("WARNING: Could not parse a numeric score from judge output.")
            print(f"Raw output: {judge_raw}")


if __name__ == "__main__":
    main()
