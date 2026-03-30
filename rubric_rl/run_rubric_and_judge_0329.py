#!/usr/bin/env python3
"""
Standalone script to generate a rubric from a question and score one or more answers using
the frozen judge model.

Pipeline:
  1. Load the rubric generator model (default: stellalisy/rubric_generator_v0_0302) via vLLM
  2. Generate a rubric for the given question
  3. Load the frozen judge model (default: Qwen/Qwen3-1.7B) via vLLM
  4. Score the provided answer(s) against the rubric

Usage:
  # Generate rubric + judge an answer (two models)
  python scripts/run_rubric_and_judge.py \
      --question "Explain the theory of relativity." \
      --answer "E=mc^2 describes mass-energy equivalence."

  # Generate rubric + judge two answers in one run
  python scripts/run_rubric_and_judge.py \
      --question "Explain the theory of relativity." \
      --answer_a "E=mc^2 describes mass-energy equivalence." \
      --answer_b "It says everything is relative."

  python scripts/run_rubric_and_judge.py \
      --rubric_model Qwen/Qwen3-8B \
      --question "Explain the theory of relativity." \
      --answer "E=mc^2 describes mass-energy equivalence."
    
  python scripts/run_rubric_and_judge.py \
      --rubric_model Qwen/Qwen3-8B \
      --question "What barriers and challenges do sales engineers face to achieve their goals?" \
      --answer_a "Organizational & Cross-Functional Barriers, Technical & Skill Development Challenges, Market & Buyer-Related Obstacles" \
      --answer_b "Some barriers include they don't have goals."

  python scripts/run_rubric_and_judge.py \
      --rubric_model Qwen/Qwen3-8B \
      --question "A study of the Moodle Learning Management system reveals that the average response time has an exponential distribution with a mean of 3 seconds. (a) What is the probability the response time exceeds 5 seconds? (b) Given that the response time exceeds 5 seconds, what is the probability the response time exceeds 10 seconds?" \
      --answer_a "Let me help you solve this step by step.\n\nFor an exponential distribution, if μ is the mean, then:\n* The rate parameter λ = 1/μ\n* The PDF is f(x) = λe^(-λx)\n* The CDF is F(x) = 1 - e^(-λx)\n* The probability P(X > x) = e^(-λx)\n\nGiven:\n* Mean (μ) = 3 seconds\n* Therefore, λ = 1/3\n\n(a) To find P(X > 5):\n\nP(X > 5) = e^(-λ5)\n= e^(-(1/3)×5)\n= e^(-5/3)\n= e^(-1.667)\n≈ 0.189 or about 18.9%\n\n(b) To find P(X > 10 | X > 5):\n\nUsing the memoryless property of exponential distributions:\nP(X > 10 | X > 5) = P(X > 5)\n\nOr we can solve it using the definition of conditional probability:\nP(X > 10 | X > 5) = P(X > 10) / P(X > 5)\n= e^(-10/3) / e^(-5/3)\n= e^(-5/3)\n= e^(-1.667)\n≈ 0.189 or about 18.9%\n\nBoth approaches give us the same answer due to the memoryless property of exponential distributions.\n\nTherefore:\n(a) The probability the response time exceeds 5 seconds is 0.189 or 18.9%\n(b) Given the response time exceeds 5 seconds, the probability it exceeds 10 seconds is 0.189 or 18.9%" \
      --answer_b "To solve the problem, we need to use the properties of the exponential distribution. The probability density function (pdf) of an exponential distribution is given by:\n\\[ f(x; \\lambda) = \\lambda e^{-\\lambda x} \\]\nwhere \\(\\lambda\\) is the rate parameter, and the mean of the distribution is given by \\( \\frac{1}{\\lambda} \\).\n\nGiven that the mean response time is 3 seconds, we have:\n\\[ \\frac{1}{\\lambda} = 3 \\implies \\lambda = \\frac{1}{3} \\]\n\n(a) The probability that the response time exceeds 5 seconds is:\n\\[ P(X > 5) = \\int_{5}^{\\infty} \\frac{1}{3} e^{-\\frac{1}{3} x} \\, dx \\]\n\nTo solve this integral, we use the fact that the integral of \\( e^{-ax} \\) from \\(a\\) to \\(\\infty\\) is \\( \\frac{e^{-ax}}{a} \\) evaluated from \\(a\\) to \\(\\infty\\):\n\\[ P(X > 5) = \\left[ -e^{-\\frac{1}{3} x} \\right]_{5}^{\\infty} = 0 - \\left( -e^{-\\frac{5}{3}} \\right) = e^{-\\frac{5}{3}} \\]\n\nUsing a calculator, we find:\n\\[ e^{-\\frac{5}{3}} \\approx 0.2183 \\]\n\nTherefore, the probability that the response time exceeds 5 seconds is:\n\\[ \\boxed{0.2183} \\]\n\n(b) The probability that the response time exceeds 10 seconds given that it exceeds 5 seconds is a conditional probability. This can be written as:\n\\[ P(X > 10 \\mid X > 5) = \\frac{P(X > 10)}{P(X > 5)} \\]\n\nWe already know \\( P(X > 5) = e^{-\\frac{5}{3}} \\). Now, we need to find \\( P(X > 10) \\):\n\\[ P(X > 10) = \\int_{10}^{\\infty} \\frac{1}{3} e^{-\\frac{1}{3} x} \\, dx = \\left[ -e^{-\\frac{1}{3} x} \\right]_{10}^{\\infty} = 0 - \\left( -e^{-\\frac{10}{3}} \\right) = e^{-\\frac{10}{3}} \\]\n\nUsing a calculator, we find:\n\\[ e^{-\\frac{10}{3}} \\approx 0.0301 \\]\n\nNow, we can calculate the conditional probability:\n\\[ P(X > 10 \\mid X > 5) = \\frac{e^{-\\frac{10}{3}}}{e^{-\\frac{5}{3}}} = e^{-\\frac{10}{3} + \\frac{5}{3}} = e^{-\\frac{5}{3}} \\]\n\nUsing a calculator again, we find:\n\\[ e^{-\\frac{5}{3}} \\approx 0.2183 \\]\n\nTherefore, the probability that the response time exceeds 10 seconds given that it exceeds 5 seconds is:\n\\[ \\boxed{0.2183} \\]"

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_instruct.search_rewards.utils.rubric_chat_templates import format_messages


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a rubric and/or judge one or more answers using vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--question", required=True, help="The question to generate a rubric for / judge against.")
    p.add_argument("--answer", default=None, help="Single answer to evaluate. If omitted, only the rubric is generated unless --answer_a/--answer_b are provided.")
    p.add_argument("--answer_a", default=None, help="First answer in a two-answer evaluation run.")
    p.add_argument("--answer_b", default=None, help="Second answer in a two-answer evaluation run.")
    p.add_argument("--rubric", default=None, help="Pre-supplied rubric text. Skips rubric generation when provided.")

    # Models
    p.add_argument("--rubric_model", default="stellalisy/rubric_generator_v0_0302",
                   help="HF model ID for rubric generation.")
    p.add_argument("--judge_model", default="Qwen/Qwen3-1.7B",
                   help="HF model ID for the frozen judge.")

    # Generation knobs
    p.add_argument("--rubric_style", choices=["standard", "v0", "v3", "correctness"], default="v3",
                   help="Rubric generation prompt style: v3 (default, dealbreaker), standard (v2), v0 (original), correctness (checklist).")
    p.add_argument("--rubric_system_prompt", default=None,
                   help="Custom system prompt for rubric generation. Overrides --rubric_style when set.")
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


def collect_answers(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """Normalize single-answer and two-answer CLI inputs."""
    pair_supplied = args.answer_a is not None or args.answer_b is not None

    if args.answer is not None and pair_supplied:
        parser.error("Use either --answer or the pair --answer_a/--answer_b, not both.")

    if pair_supplied:
        if args.answer_a is None or args.answer_b is None:
            parser.error("When using the two-answer mode, both --answer_a and --answer_b are required.")
        return [("Answer A", args.answer_a), ("Answer B", args.answer_b)]

    if args.answer is not None:
        return [("Answer", args.answer)]

    return []


def judge_answer(
    *,
    question: str,
    rubric_for_judge: str,
    answer_label: str,
    answer_text: str,
    judge_mode: str,
    judge_model: str,
    judge_engine: vllm.LLM,
    judge_tokenizer,
    judge_temperature: float,
    judge_max_tokens: int,
    judge_enable_thinking: bool,
) -> dict:
    """Run the judge on a single answer and print the result block."""
    template = "judge_binary" if judge_mode == "binary" else "judge"
    messages = format_messages(
        template,
        {
            "question": question,
            "rubric": rubric_for_judge,
            "answer": answer_text,
        },
        tokenize=False,
    )

    print("=" * 60)
    print(f"JUDGE EVALUATION: {answer_label}")
    print("=" * 60)
    print(f"Model : {judge_model}")
    print(f"Mode  : {judge_mode}")
    print(f"{answer_label}: {answer_text[:200]}{'...' if len(answer_text) > 200 else ''}\n")

    judge_output, judge_raw = generate(
        judge_engine,
        judge_tokenizer,
        messages,
        judge_temperature,
        judge_max_tokens,
        disable_thinking=not judge_enable_thinking,
    )

    print("--- Raw Judge Output ---")
    print(judge_raw)
    print("--- End Raw Output ---\n")
    print("--- Cleaned Judge Output (thinking stripped) ---")
    print(judge_output)
    print("--- End Cleaned Output ---\n")

    if judge_mode == "binary":
        result = parse_binary_judgment(judge_output)
        print(f"Verdict: {result['verdict']}")
        return result

    result = parse_judge_score(judge_output)
    if result["score"] is not None:
        print(f"Score : {result['score']:.2f}")
        print(f"Reason: {result['reasoning']}")
    else:
        print("WARNING: Could not parse a numeric score from judge output.")
        print(f"Raw output: {judge_raw}")
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    answers = collect_answers(args, parser)

    need_rubric_gen = args.rubric is None
    need_judge = bool(answers)
    same_model = need_rubric_gen and need_judge and args.rubric_model == args.judge_model

    rubric_text = args.rubric

    # --- Step 1: Rubric generation ---
    if need_rubric_gen:
        rubric_max_len = args.max_model_len or 16384
        rubric_engine = create_engine(args.rubric_model, args.tensor_parallel_size,
                                      args.gpu_memory_utilization, rubric_max_len, args.trust_remote_code)
        rubric_tokenizer = AutoTokenizer.from_pretrained(args.rubric_model, trust_remote_code=True)

        if args.rubric_system_prompt is not None:
            messages = [
                {"role": "system", "content": args.rubric_system_prompt},
                {"role": "user", "content": args.question},
            ]
        else:
            style_to_template = {
                "standard": "rubric_generation",
                "v0": "rubric_generation_v0",
                "v3": "rubric_generation_v3",
                "correctness": "rubric_generation_correctness",
            }
            template = style_to_template[args.rubric_style]
            messages = format_messages(template, {"question": args.question}, tokenize=False)

        print("\n" + "=" * 60)
        print("RUBRIC GENERATION")
        print("=" * 60)
        print(f"Model : {args.rubric_model}")
        print(f"Style : {'custom' if args.rubric_system_prompt else args.rubric_style}")
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
        print("No answer input provided; stopping after rubric generation.")
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

    results: list[tuple[str, dict]] = []
    for answer_label, answer_text in answers:
        result = judge_answer(
            question=args.question,
            rubric_for_judge=rubric_for_judge,
            answer_label=answer_label,
            answer_text=answer_text,
            judge_mode=args.judge_mode,
            judge_model=args.judge_model,
            judge_engine=judge_engine,
            judge_tokenizer=judge_tokenizer,
            judge_temperature=args.judge_temperature,
            judge_max_tokens=args.judge_max_tokens,
            judge_enable_thinking=args.judge_enable_thinking,
        )
        results.append((answer_label, result))

    if len(results) == 2:
        print("\n" + "=" * 60)
        print("PAIR SUMMARY")
        print("=" * 60)
        if args.judge_mode == "binary":
            for answer_label, result in results:
                print(f"{answer_label}: {result['verdict']}")
        else:
            for answer_label, result in results:
                score = result.get("score")
                score_text = f"{score:.2f}" if score is not None else "unparsed"
                print(f"{answer_label}: {score_text}")

            score_a = results[0][1].get("score")
            score_b = results[1][1].get("score")
            if score_a is not None and score_b is not None:
                if score_a > score_b:
                    winner = results[0][0]
                elif score_b > score_a:
                    winner = results[1][0]
                else:
                    winner = "Tie"
                print(f"Higher score: {winner}")


if __name__ == "__main__":
    main()
