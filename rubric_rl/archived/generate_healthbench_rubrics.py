#!/usr/bin/env python3
"""
Batch-generate per-sample rubrics for HealthBench queries using vLLM.

Pipeline:
  1. Download / load HealthBench data (all / hard / consensus subset)
  2. Convert each prompt into a question string
  3. Use the rubric_generation template from rubric_chat_templates.py
     to format rubric-generation prompts
  4. Batch-generate rubrics with a vLLM engine
  5. Save results as JSONL  (prompt_id, question, generated_rubric, ...)

Usage (offline vLLM):
  python generate_healthbench_rubrics.py \
      --rubric_model stellalisy/rubric_generator_v0_0302 \
      --subset all \
      --output /checkpoint/dream/rulin/dr-tulu/rubric_rl/healthbench_rubrics.jsonl

  # Use correctness-focused rubric style
  python generate_healthbench_rubrics.py \
      --rubric_model stellalisy/rubric_generator_v0_0302 \
      --rubric_style correctness \
      --output /checkpoint/dream/rulin/dr-tulu/rubric_rl/healthbench_rubrics_correctness.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import vllm
from transformers import AutoTokenizer

from rubric_chat_templates import format_messages


# ── HealthBench data loading ─────────────────────────────────────────────────
HEALTHBENCH_URLS = {
    "all": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
    "consensus": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl",
}


def _get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "dr_agent" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_if_needed(url: str, cache_name: str) -> Path:
    cache_path = _get_cache_dir() / cache_name
    if not cache_path.exists():
        print(f"Downloading {url} -> {cache_path}")
        urllib.request.urlretrieve(url, cache_path)
    return cache_path


def _format_conversation(conversation: list[dict]) -> str:
    role_mapping = {"user": "Patient", "assistant": "Physician"}
    return "\n\n".join(
        f"{role_mapping.get(m['role'], m['role'])}: {m['content']}" for m in conversation
    )


def stringify_prompt(prompt: list[dict]) -> str:
    """Convert a HealthBench prompt (list of messages) into a single question string."""
    if len(prompt) == 1:
        return (
            "Can you answer the question from a patient about a medical condition "
            f"or concern they have: {prompt[0]['content']}"
        )
    elif len(prompt) > 1:
        return (
            "Here is a conversation between a patient and a physician. "
            "The patient is asking a question about a medical condition or concern they have, "
            "and in the conversation it should contain necessary background information about the patient:\n\n"
            f"{_format_conversation(prompt[:-1])}\n\n"
            f"Can you search for needed information and answer the patient's question: {prompt[-1]['content']}"
        )
    else:
        raise ValueError("Data error: prompt must have at least one element.")


def load_healthbench(
    subset: str = "all",
    local_path: str | None = None,
    num_examples: int | None = None,
) -> list[dict]:
    """Load HealthBench examples and attach a 'question' field."""
    if local_path and Path(local_path).exists():
        path = Path(local_path)
    else:
        url = HEALTHBENCH_URLS[subset]
        path = _download_if_needed(url, f"healthbench_{subset}.jsonl")

    with open(path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    for ex in examples:
        ex["question"] = stringify_prompt(ex["prompt"])

    if num_examples is not None and num_examples > 0:
        examples = examples[:num_examples]

    print(f"Loaded {len(examples)} HealthBench examples (subset={subset})")
    return examples


# ── vLLM helpers ─────────────────────────────────────────────────────────────

def _supports_enable_thinking(tokenizer) -> bool:
    return (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
        and "enable_thinking" in tokenizer.chat_template
    )


def _strip_thinking_block(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return cleaned.strip()


def create_engine(
    model: str,
    tp: int,
    gpu_mem: float,
    max_len: int | None,
    trust_remote_code: bool,
) -> vllm.LLM:
    kwargs = dict(
        model=model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=trust_remote_code,
    )
    if max_len is not None:
        kwargs["max_model_len"] = max_len
    print(f"Loading model: {model}  (tp={tp}, gpu_mem={gpu_mem}, max_len={max_len})")
    return vllm.LLM(**kwargs)


def batch_generate(
    engine: vllm.LLM,
    tokenizer,
    prompts_text: list[str],
    temperature: float,
    max_tokens: int,
    batch_size: int = 256,
) -> list[tuple[str, str]]:
    """Generate completions in batches. Returns list of (cleaned, raw) tuples."""
    params = vllm.SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
    all_results: list[tuple[str, str]] = []

    for start in range(0, len(prompts_text), batch_size):
        batch = prompts_text[start : start + batch_size]
        print(f"  Generating batch {start // batch_size + 1} "
              f"({start+1}-{min(start+len(batch), len(prompts_text))} of {len(prompts_text)})")
        outputs = engine.generate(batch, params)
        for out in outputs:
            raw = out.outputs[0].text.strip()
            cleaned = _strip_thinking_block(raw)
            all_results.append((cleaned, raw))

    return all_results


# ── Main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-generate per-sample rubrics for HealthBench using vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    p.add_argument("--subset", choices=["all", "hard", "consensus"], default="all",
                   help="HealthBench subset to use.")
    p.add_argument("--local_path", default=None,
                   help="Optional local path to a pre-downloaded healthbench JSONL.")
    p.add_argument("--num_examples", type=int, default=None,
                   help="Limit to first N examples (for debugging).")
    p.add_argument("--output", required=True,
                   help="Path to write output JSONL with generated rubrics.")

    # Model
    p.add_argument("--rubric_model", default="stellalisy/rubric_generator_v0_0302",
                   help="HF model ID (or local path) for rubric generation.")
    p.add_argument("--rubric_style", choices=["standard", "correctness"], default="standard",
                   help="Rubric generation prompt style from rubric_chat_templates.")

    # Generation
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Number of prompts to send to vLLM per batch.")
    p.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable thinking for rubric generator (default: True).")

    # vLLM
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=16384,
                   help="Max sequence length for the model.")
    p.add_argument("--trust_remote_code", action="store_true", default=True)

    return p


def main():
    args = build_parser().parse_args()

    # 1. Load data
    examples = load_healthbench(
        subset=args.subset,
        local_path=args.local_path,
        num_examples=args.num_examples,
    )

    # 2. Build rubric-generation prompts using rubric_chat_templates
    template_name = (
        "rubric_generation_correctness" if args.rubric_style == "correctness"
        else "rubric_generation"
    )
    print(f"Using template: {template_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.rubric_model, trust_remote_code=True)

    template_kwargs: dict = dict(tokenize=False, add_generation_prompt=True)
    disable_thinking = not args.enable_thinking
    if _supports_enable_thinking(tokenizer) and disable_thinking:
        template_kwargs["enable_thinking"] = False

    prompts_text = []
    for ex in examples:
        messages = format_messages(template_name, {"question": ex["question"]}, tokenize=False)
        prompt_str = tokenizer.apply_chat_template(messages, **template_kwargs)
        prompts_text.append(prompt_str)

    print(f"Formatted {len(prompts_text)} prompts. First prompt length: {len(prompts_text[0])} chars")

    # 3. Load model & generate
    engine = create_engine(
        args.rubric_model,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        args.max_model_len,
        args.trust_remote_code,
    )

    results = batch_generate(
        engine, tokenizer, prompts_text,
        args.temperature, args.max_tokens, args.batch_size,
    )

    # 4. Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex, (cleaned, raw) in zip(examples, results):
            record = {
                "prompt_id": ex["prompt_id"],
                "question": ex["question"],
                "prompt": ex["prompt"],
                "generated_rubric": cleaned,
                "generated_rubric_raw": raw,
                "original_rubrics": ex.get("rubrics", []),
                "rubric_style": args.rubric_style,
                "rubric_model": args.rubric_model,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone! Wrote {len(results)} rubrics to {output_path}")

    # Quick stats
    empty_count = sum(1 for c, _ in results if not c.strip())
    print(f"  Empty rubrics: {empty_count}/{len(results)}")
    avg_len = sum(len(c) for c, _ in results) / max(len(results), 1)
    print(f"  Avg rubric length: {avg_len:.0f} chars")


if __name__ == "__main__":
    main()

