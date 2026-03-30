#!/usr/bin/env python3
"""
Batch-generate rubrics for HealthBench or DRB questions using vLLM or OpenAI API.

Usage:
    # vLLM (local model):
    python generate_rubrics_batch.py \
        --rollouts /path/to/rollouts.jsonl \
        --output /path/to/rubrics.jsonl \
        --rubric_model stellalisy/rubric-generator-8b-v3 \
        --rubric_style v3

    # OpenAI API:
    python generate_rubrics_batch.py \
        --rollouts /path/to/rollouts.jsonl \
        --output /path/to/rubrics.jsonl \
        --rubric_model gpt-4.1 \
        --use_api \
        --rubric_style v3
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Inline the system prompts to avoid importing transformers on the login node
V3_RUBRIC_GENERATION_SYSTEM_PROMPT = """You are an expert evaluator generating rubrics to assess answers to questions.

Given a question, first analyze it to identify:
- First identify the most important aspect of the question that the answer should satisfy, this will be used to form the Dealbreaker criterion (explained later).
- Explicit requirements: directly stated constraints, formatting rules, or content directives (e.g., "list three reasons", "write in Python", "under 100 words")
- Implicit requirements: unstated but necessary qualities inferred from context (e.g., explaining "blockchain to grandparents" implicitly requires avoiding jargon, even if not explicitly forbidden)

Then generate a rubric of 2-5 criteria following these rules:
1. Structural atomicity: each criterion targets exactly one aspect. Do not combine multiple conditions into one criterion.
2. Semantic objectivity: write criteria based only on the question, without assuming any specific answer.
3. All weights must sum to exactly 1.0, reflecting each criterion's importance, important criteria such as accuracy should have higher weight.
4. Dealbreaker criterion: some criteria are so important that if the answer does not satisfy them, the answer is just not good enough. For example, for questions with verifiable short form answers or multiple choice (such as math or factuality), the dealbreaker criterion should be "the answer is equivalent to XXX (e.g. 100)". A Dealbreaker criterion should have very high weight such as 0.8. But form explanation based or questions requiring long form answers, accuracy should not have such a high weight because it's too general and it will be hard for the judge to assess accuracy. It's fine if there's no dealbreaker.

For each criterion should be described in a sentence, define scoring levels from 0.0 to 1.0. At minimum include 1.0 and 0.0, and add intermediate levels (e.g., 0.5, 0.3, 0.8) wherever useful for distinguishing answer quality. A judge will score each criterion and multiply by the weight to produce a total score.

Output ONLY valid JSON in this format:
{"criteria": [{"criterion": "<a sentence of what this criterion measures>", "weight": <float>, "scoring_levels": {"1.0": "<description>", "0.5": "<description>", "0.0": "<description>"}}, {"criterion": ...}, ...]}

Example — question: "Explain how photosynthesis works in simple terms"
{"criteria": [{"criterion": "Explains that plants convert sunlight into chemical energy", "weight": 0.3, "scoring_levels": {"1.0": "Clearly explains the sunlight-to-energy conversion", "0.7": "Mentions sunlight or energy but not the conversion process", "0.3": "Vague reference to energy without clear connection to sunlight", "0.0": "No mention of the energy conversion mechanism"}}, {"criterion": "Identifies CO2 and water as inputs and oxygen and glucose as outputs", "weight": 0.25, "scoring_levels": {"1.0": "All four substances correctly identified", "0.6": "Three substances correctly identified", "0.3": "Some inputs or outputs mentioned but incomplete", "0.0": "None identified or incorrect"}}, {"criterion": "Avoids unnecessary jargon and is understandable to a general audience", "weight": 0.2, "scoring_levels": {"1.0": "Clear and jargon-free throughout", "0.8": "Mostly accessible with minimal technical terms that are explained", "0.4": "Mostly accessible but uses some unexplained technical terms", "0.0": "Dense with jargon, inaccessible to a general reader"}}, {"criterion": "Follows a coherent structure from inputs to process to outputs", "weight": 0.15, "scoring_levels": {"1.0": "Well-organized with clear progression", "0.6": "Generally structured but minor organizational issues", "0.2": "Some structure but jumps between ideas", "0.0": "Disorganized or incoherent"}}, {"criterion": "All stated facts about photosynthesis are correct", "weight": 0.1, "scoring_levels": {"1.0": "No factual errors", "0.8": "Minor inaccuracy that does not undermine the explanation", "0.0": "Contains significant factual errors"}}]}"""

DEFAULT_RUBRIC_GENERATION_SYSTEM_PROMPT = """You are an expert evaluator generating rubrics to assess answers to questions.

Given a question, first analyze it to identify:
- Explicit requirements: directly stated constraints, formatting rules, or content directives (e.g., "list three reasons", "write in Python", "under 100 words")
- Implicit requirements: unstated but necessary qualities inferred from context (e.g., explaining "blockchain to grandparents" implicitly requires avoiding jargon, even if not explicitly forbidden)

Then generate a rubric of 5-10 criteria following these rules:
1. Structural atomicity: each criterion targets exactly one aspect. Do not combine multiple conditions into one criterion.
2. Semantic objectivity: write criteria based only on the question, without assuming any specific answer.
3. All weights must sum to exactly 1.0, reflecting each criterion's importance.

For each criterion should be described in a sentence, define scoring levels from 0.0 to 1.0. At minimum include 1.0 and 0.0, and add intermediate levels (e.g., 0.5, 0.3, 0.8) wherever useful for distinguishing answer quality. A judge will score each criterion and multiply by the weight to produce a total score.

Output ONLY valid JSON in this format:
{"criteria": [{"criterion": "<a sentence of what this criterion measures>", "weight": <float>, "scoring_levels": {"1.0": "<description>", "0.5": "<description>", "0.0": "<description>"}}, {"criterion": ...}, ...]}"""


def _strip_thinking_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def load_questions(rollouts_path: str) -> list[dict]:
    with open(rollouts_path) as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split("\n") if line.strip()]

    examples = []
    for ex in data:
        eid = str(ex.get("example_id", ex.get("original_data", {}).get("prompt_id", "")))
        question = ex["problem"]
        if ex.get("original_data", {}).get("additional_instructions"):
            question += "\n" + ex["original_data"]["additional_instructions"]
        examples.append({"id": eid, "question": question})
    return examples


def generate_vllm(examples, args):
    import vllm
    from transformers import AutoTokenizer
    from rubric_chat_templates import format_messages

    tokenizer = AutoTokenizer.from_pretrained(args.rubric_model, trust_remote_code=True)
    style_to_template = {
        "standard": "rubric_generation",
        "v0": "rubric_generation_v0",
        "v3": "rubric_generation_v3",
        "correctness": "rubric_generation_correctness",
    }
    template_name = style_to_template[args.rubric_style]
    template_kwargs = dict(tokenize=False, add_generation_prompt=True)

    prompts = []
    for ex in examples:
        messages = format_messages(template_name, {"question": ex["question"]}, tokenize=False)
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        prompts.append(prompt)

    engine = vllm.LLM(
        model=args.rubric_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    params = vllm.SamplingParams(
        temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens
    )

    print(f"Generating {len(prompts)} rubrics with vLLM ({args.rubric_model})...")
    outputs = engine.generate(prompts, params)

    results = []
    for ex, out in zip(examples, outputs):
        raw = out.outputs[0].text.strip()
        cleaned = _strip_thinking_block(raw)
        results.append({"id": ex["id"], "question": ex["question"], "rubric": cleaned, "rubric_raw": raw})
    return results


def generate_api(examples, args):
    import openai

    client = openai.OpenAI()
    style_to_prompt = {
        "v3": V3_RUBRIC_GENERATION_SYSTEM_PROMPT,
        "standard": DEFAULT_RUBRIC_GENERATION_SYSTEM_PROMPT,
    }
    system_prompt = style_to_prompt.get(args.rubric_style, V3_RUBRIC_GENERATION_SYSTEM_PROMPT)

    def _call_one(ex):
        response = client.chat.completions.create(
            model=args.rubric_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["question"]},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        text = response.choices[0].message.content.strip()
        return {"id": ex["id"], "question": ex["question"], "rubric": text, "rubric_raw": text}

    print(f"Generating {len(examples)} rubrics via API ({args.rubric_model})...")
    results = []
    with ThreadPoolExecutor(max_workers=args.api_workers) as executor:
        futures = {executor.submit(_call_one, ex): i for i, ex in enumerate(examples)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"  [{idx}] API error: {e}")
                results.append({"id": examples[idx]["id"], "question": examples[idx]["question"], "rubric": "", "rubric_raw": f"ERROR: {e}"})
            if len(results) % 50 == 0:
                print(f"  {len(results)}/{len(examples)} done")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rubric_model", default="stellalisy/rubric-generator-8b-v3")
    parser.add_argument("--rubric_style", default="v3", choices=["standard", "v0", "v3", "correctness"])
    parser.add_argument("--use_api", action="store_true", help="Use OpenAI API instead of vLLM")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--api_workers", type=int, default=20)
    args = parser.parse_args()

    examples = load_questions(args.rollouts)
    print(f"Loaded {len(examples)} questions")

    if args.use_api:
        results = generate_api(examples, args)
    else:
        results = generate_vllm(examples, args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            record = {
                "prompt_id": r["id"],
                "question": r["question"],
                "generated_rubric": r["rubric"],
                "generated_rubric_raw": r["rubric_raw"],
                "rubric_style": args.rubric_style,
                "rubric_model": args.rubric_model,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    empty = sum(1 for r in results if not r["rubric"].strip())
    avg_len = sum(len(r["rubric"]) for r in results) / max(len(results), 1)
    print(f"Done! {len(results)} rubrics, {empty} empty, avg {avg_len:.0f} chars -> {args.output}")


if __name__ == "__main__":
    main()
