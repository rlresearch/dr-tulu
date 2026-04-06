#!/usr/bin/env python3
"""
Generate rubrics for Deep Research Bench questions using the rubric generator model.

Usage:
    python generate_drb_rubrics.py \
        --rollouts /path/to/deep_research_bench.jsonl \
        --output /path/to/drb_rubrics.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import vllm
from transformers import AutoTokenizer

from rubric_chat_templates import format_messages


def _supports_enable_thinking(tokenizer) -> bool:
    return (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
        and "enable_thinking" in tokenizer.chat_template
    )


def _strip_thinking_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True, help="Path to DRB rollouts JSONL (to extract questions)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rubric_model", default="stellalisy/rubric_generator_v0_0302")
    parser.add_argument("--rubric_style", default="standard", choices=["standard", "correctness"])
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    args = parser.parse_args()

    with open(args.rollouts) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(examples)} DRB examples")

    tokenizer = AutoTokenizer.from_pretrained(args.rubric_model, trust_remote_code=True)
    template_name = "rubric_generation_correctness" if args.rubric_style == "correctness" else "rubric_generation"

    template_kwargs = dict(tokenize=False, add_generation_prompt=True)

    prompts = []
    for ex in examples:
        question = ex["problem"]
        if ex.get("original_data", {}).get("additional_instructions"):
            question += "\n" + ex["original_data"]["additional_instructions"]
        messages = format_messages(template_name, {"question": question}, tokenize=False)
        prompt_str = tokenizer.apply_chat_template(messages, **template_kwargs)
        prompts.append(prompt_str)

    print(f"Formatted {len(prompts)} prompts")

    engine = vllm.LLM(
        model=args.rubric_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    params = vllm.SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens)

    print("Generating rubrics...")
    outputs = engine.generate(prompts, params)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex, out in zip(examples, outputs):
            raw = out.outputs[0].text.strip()
            cleaned = _strip_thinking_block(raw)
            record = {
                "prompt_id": str(ex["example_id"]),
                "question": ex["problem"],
                "generated_rubric": cleaned,
                "generated_rubric_raw": raw,
                "rubric_style": args.rubric_style,
                "rubric_model": args.rubric_model,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    empty = sum(1 for o in outputs if not _strip_thinking_block(o.outputs[0].text.strip()))
    avg_len = sum(len(_strip_thinking_block(o.outputs[0].text.strip())) for o in outputs) / len(outputs)
    print(f"Done! {len(outputs)} rubrics, {empty} empty, avg length {avg_len:.0f} chars")


if __name__ == "__main__":
    main()
