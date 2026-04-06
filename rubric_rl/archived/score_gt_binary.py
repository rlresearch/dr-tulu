#!/usr/bin/env python3
"""Score rollouts against GT expert rubrics using GPT-4.1 with binary per-criterion grading.
Same method as HealthBench: for each criterion, ask if met (true/false), sum weights."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

GRADER_TEMPLATE = """You are an expert evaluator. Below is a question and a response, followed by a rubric criterion. Determine if the criterion is met based on the response.

IMPORTANT: Output JSON with two fields:
- "criteria_met": true or false
- "explanation": a brief explanation

Question: {question}

Response: {response}

Criterion [{weight}]: {criterion}

Output:"""


def score_one(question, answer, rubric_json, client, model):
    rubric = json.loads(rubric_json)
    criteria = rubric.get("criteria", [])
    if not criteria:
        return None

    total_weight = 0.0
    achieved = 0.0

    for crit in criteria:
        w = crit["weight"]
        prompt = GRADER_TEMPLATE.format(
            question=question,
            response=answer[:20000],
            weight=w,
            criterion=crit["criterion"],
        )
        for _ in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=500,
                )
                text = resp.choices[0].message.content
                js, je = text.find("{"), text.rfind("}")
                if js != -1 and je != -1:
                    result = json.loads(text[js:je+1])
                    if "criteria_met" in result and isinstance(result["criteria_met"], bool):
                        if w > 0:
                            total_weight += w
                            if result["criteria_met"]:
                                achieved += w
                        else:
                            total_weight += abs(w)
                            if not result["criteria_met"]:
                                achieved += abs(w)
                        break
            except Exception as e:
                pass

    if total_weight == 0:
        return None
    return achieved / total_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--rubrics", required=True)
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
