#!/usr/bin/env python3
"""Score DRB rollouts with GPT-4.1 as expert judge (RACE-style quality assessment).
Evaluates on 4 dimensions: comprehensiveness, insight, instruction_following, readability."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

JUDGE_PROMPT = """You are an expert evaluator assessing the quality of a deep research report.

Question/Task:
{question}

{additional_instructions}

Report to evaluate:
{response}

Rate the report on each dimension from 0.0 to 1.0:
1. **Comprehensiveness**: Does it cover all important aspects of the topic with sufficient depth?
2. **Insight**: Does it provide novel analysis, synthesis, or non-obvious connections beyond surface-level facts?
3. **Instruction Following**: Does it follow the task instructions (structure, citations, data-driven, etc.)?
4. **Readability**: Is it well-organized, clearly written, with logical flow?

Output ONLY valid JSON:
{{"comprehensiveness": <float>, "insight": <float>, "instruction_following": <float>, "readability": <float>, "overall": <float>, "reasoning": "<brief explanation>"}}

The overall score should be a weighted average emphasizing comprehensiveness (0.3) and insight (0.3), with instruction_following (0.2) and readability (0.2)."""


def score_one(rollout, client, model):
    question = rollout["problem"]
    additional = rollout.get("original_data", {}).get("additional_instructions", "")
    response = rollout["final_response"]

    prompt = JUDGE_PROMPT.format(
        question=question,
        additional_instructions=f"Additional instructions: {additional}" if additional else "",
        response=response[:60000],
    )

    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
            )
            text = resp.choices[0].message.content
            js, je = text.find("{"), text.rfind("}")
            if js != -1 and je != -1:
                data = json.loads(text[js:je+1])
                if "overall" in data:
                    return data
        except Exception as e:
            print(f"  Error: {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--max-workers", type=int, default=20)
    args = parser.parse_args()

    client = openai.OpenAI()

    with open(args.rollouts) as f:
        data = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(data)} rollouts")

    results = {}
    def _do(r):
        eid = str(r.get("example_id", r.get("original_data", {}).get("prompt_id", "")))
        score = score_one(r, client, args.model)
        return eid, score

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(_do, r) for r in data]
        done = 0
        for f in as_completed(futures):
            eid, score = f.result()
            results[eid] = score
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(data)}")

    valid = sum(1 for s in results.values() if s is not None)
    print(f"Scored {valid}/{len(results)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
