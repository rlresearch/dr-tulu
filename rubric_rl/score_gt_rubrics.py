#!/usr/bin/env python3
"""Score rollouts with ground-truth (expert) HealthBench rubrics using GPT grader."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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


def score_one(rollout, grader):
    rubrics = rollout["original_data"]["rubrics"]
    total_pos = sum(r["points"] for r in rubrics if r["points"] > 0)
    if total_pos == 0:
        return None

    prompt_msgs = rollout["original_data"].get(
        "prompt", [{"role": "user", "content": rollout["problem"]}]
    )
    convo_str = "\n\n".join(
        f"{m['role']}: {m['content']}"
        for m in prompt_msgs + [{"role": "assistant", "content": rollout["final_response"]}]
    )

    achieved = 0.0
    for item in rubrics:
        rubric_str = f"[{item['points']}] {item['criterion']}"
        prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace("<<rubric_item>>", rubric_str)
        for _ in range(3):
            resp = grader([{"content": prompt, "role": "user"}])
            text = resp.response_text
            js, je = text.find("{"), text.rfind("}")
            if js != -1 and je != -1:
                try:
                    result = json.loads(text[js:je+1])
                    if "criteria_met" in result and isinstance(result["criteria_met"], bool):
                        if result["criteria_met"]:
                            achieved += item["points"]
                        break
                except json.JSONDecodeError:
                    continue
    return achieved / total_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grader-model", default="gpt-4.1-mini")
    parser.add_argument("--max-workers", type=int, default=20)
    args = parser.parse_args()

    with open(args.rollouts) as f:
        content = f.read().strip()
        data = json.loads(content) if content.startswith("[") else [json.loads(l) for l in content.split("\n") if l.strip()]
    print(f"Loaded {len(data)} rollouts")

    grader = ChatCompletionSampler(model=args.grader_model, system_message=OPENAI_SYSTEM_MESSAGE_API, max_tokens=1000, temperature=0)

    results = {}
    def _do(rollout):
        pid = rollout["original_data"]["prompt_id"]
        score = score_one(rollout, grader)
        return pid, score

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(_do, r) for r in data]
        done = 0
        for f in as_completed(futures):
            pid, score = f.result()
            results[pid] = score
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(data)}")

    valid = sum(1 for s in results.values() if s is not None)
    print(f"Scored {valid}/{len(results)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
