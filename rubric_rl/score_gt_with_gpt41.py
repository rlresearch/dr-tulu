#!/usr/bin/env python3
"""Score rollouts against GT expert rubrics using GPT-4.1 as judge.
Uses the same judge prompt template as the frozen Qwen3-1.7B judge."""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

JUDGE_SYSTEM = "You are an expert evaluator judging answers based on a rubric."

JUDGE_USER = """Question: {question}

Rubric: {rubric}

Answer to evaluate: {answer}

Evaluate the answer against the rubric. For each criterion, decide how well the answer satisfies it (0.0 = not at all, 1.0 = fully), then multiply by the criterion's weight. Sum the weighted scores to get the total (must be between 0.0 and 1.0).

Output ONLY valid JSON:
{{"reasoning": "<evaluate each criterion, give satisfaction * weight, then sum>", "score": <float 0.0-1.0>}}

Your evaluation:"""


def score_one(question, answer, rubric_text, client, model):
    prompt = JUDGE_USER.format(question=question, rubric=rubric_text, answer=answer[:30000])
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=2000,
            )
            text = resp.choices[0].message.content
            js, je = text.find("{"), text.rfind("}")
            if js != -1 and je != -1:
                data = json.loads(text[js:je+1])
                if "score" in data:
                    return float(data["score"])
        except Exception as e:
            print(f"  Error: {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--rubrics", required=True, help="GT rubrics in same JSONL format as synthetic")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--max-workers", type=int, default=20)
    args = parser.parse_args()

    client = openai.OpenAI()

    with open(args.rollouts) as f:
        content = f.read().strip()
        rollouts = json.loads(content) if content.startswith("[") else [json.loads(l) for l in content.split("\n") if l.strip()]

    rollout_map = {}
    for r in rollouts:
        pid = str(r.get("example_id", r.get("original_data", {}).get("prompt_id", "")))
        rollout_map[pid] = r

    with open(args.rubrics) as f:
        rubrics = {r["prompt_id"]: r for r in (json.loads(l) for l in f)}

    overlap = sorted(set(rollout_map) & set(rubrics))
    print(f"Matched {len(overlap)} examples")

    results = {}
    def _do(pid):
        r = rollout_map[pid]
        rb = rubrics[pid]
        score = score_one(rb["question"], r["final_response"], rb["generated_rubric"], client, args.model)
        return pid, score

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(_do, pid) for pid in overlap]
        done = 0
        for f in as_completed(futures):
            pid, score = f.result()
            results[pid] = score
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(overlap)}")

    valid = sum(1 for s in results.values() if s is not None)
    print(f"Scored {valid}/{len(results)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
