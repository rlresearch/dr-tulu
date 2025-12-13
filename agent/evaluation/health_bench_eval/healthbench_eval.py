"""
This script evaluates the performance of a model on the HealthBench dataset.

To run HealthBench, HealthBench Consensus, or HealthBench Hard, use the simple-evals script:
- `python -m simple-evals.simple_evals --eval=healthbench --model=gpt-4.1`
- `python -m simple-evals.simple_evals --eval=healthbench_consensus --model=gpt-4.1`
- `python -m simple-evals.simple_evals --eval=healthbench_hard --model=gpt-4.1`

You can also evaluate physician ideal completions or reference completions against the HealthBench rubrics. To do so, run the following command:
- To evaluate physician ideal completions: `python -m simple-evals.healthbench_eval --run_mode=physician_completions`
- To evaluate reference model completions used by physicians: `python -m simple-evals.healthbench_eval --run_mode=physician_completion_references`
"""

import argparse
import copy
import hashlib
import json
import os
import random
import re
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from samplers import common
from samplers._types import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult
from samplers.sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)
from tqdm import tqdm

INPUT_PATH = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
INPUT_PATH_HARD = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
INPUT_PATH_CONSENSUS = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

HEALTHBENCH_HTML_JINJA = (
    common.HTML_JINJA.replace(
        "<p>Correct Answer: {{ correct_answer }}</p>\n",
        "",
    )
    + "<p>Rubrics with grades: {{ rubric_grades }}</p>"
)


def parse_json_to_dict(json_string: str) -> dict:
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {}


class RubricItem:
    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"[{self.points}] {self.criterion}"

    def to_dict(self):
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d["tags"],
        )


def calculate_score(
    rubric_items: list[RubricItem], grading_response_list: list[dict]
) -> float | None:
    total_possible_points = sum(
        rubric_item.points for rubric_item in rubric_items if rubric_item.points > 0
    )
    if total_possible_points == 0:
        # should not happen for overall score, but may happen for tags
        return None

    achieved_points = sum(
        rubric_item.points
        for rubric_item, grading_response in zip(
            rubric_items, grading_response_list, strict=True
        )
        if grading_response["criteria_met"]
    )
    overall_score = achieved_points / total_possible_points
    return overall_score


def get_usage_dict(response_usage) -> dict[str, int | None]:
    if response_usage is None:
        return {
            "input_tokens": None,
            "input_cached_tokens": None,
            "output_tokens": None,
            "output_reasoning_tokens": None,
            "total_tokens": None,
        }

    try:
        input_details = getattr(response_usage, "input_tokens_details", None)
        cached_tokens = None
        if input_details:
            if hasattr(input_details, "cached_tokens"):
                cached_tokens = input_details.cached_tokens
            elif isinstance(input_details, dict):
                cached_tokens = input_details.get("cached_tokens")

        output_details = getattr(response_usage, "output_tokens_details", None)
        reasoning_tokens = None
        if output_details:
            if hasattr(output_details, "reasoning_tokens"):
                reasoning_tokens = output_details.reasoning_tokens
            elif isinstance(output_details, dict):
                reasoning_tokens = output_details.get("reasoning_tokens")

        return {
            "input_tokens": response_usage.input_tokens,
            "input_cached_tokens": cached_tokens,
            "output_tokens": response_usage.output_tokens,
            "output_reasoning_tokens": reasoning_tokens,
            "total_tokens": response_usage.total_tokens,
        }
    except AttributeError:
        prompt_details = getattr(response_usage, "prompt_tokens_details", None)
        cached_tokens = None
        if prompt_details:
            if hasattr(prompt_details, "cached_tokens"):
                cached_tokens = prompt_details.cached_tokens
            elif isinstance(prompt_details, dict):
                cached_tokens = prompt_details.get("cached_tokens")

        completion_details = getattr(response_usage, "completion_tokens_details", None)
        reasoning_tokens = None
        if completion_details:
            if hasattr(completion_details, "reasoning_tokens"):
                reasoning_tokens = completion_details.reasoning_tokens
            elif isinstance(completion_details, dict):
                reasoning_tokens = completion_details.get("reasoning_tokens")

        return {
            "input_tokens": response_usage.prompt_tokens,
            "input_cached_tokens": cached_tokens,
            "output_tokens": response_usage.completion_tokens,
            "output_reasoning_tokens": reasoning_tokens,
            "total_tokens": response_usage.total_tokens,
        }


PHYSICIAN_COMPLETION_MODES = {
    "Group 1": {
        "description": "No reference completions were provided to the physicians.",
        "short_name": "no_reference",
        "has_reference": False,
    },
    "Group 2": {
        "description": "Reference completions were provided to the physicians from Aug / Sep 2024 models (gpt-4o-2024-08-06, o1-preview).",
        "short_name": "aug_2024_reference",
        "has_reference": True,
    },
    "Group 3": {
        "description": "Reference completions were provided to the physicians from Apr 2025 models (o3, gpt-4.1).",
        "short_name": "apr_2025_reference",
        "has_reference": True,
    },
}


def _compute_clipped_stats(
    values: list,
    stat: str,
):
    """Computes the mean (clipped to [0, 1]), bootstrap std for that mean, and n_samples for final HealthBench scoring."""
    if stat == "mean":
        return np.clip(np.mean(values), 0, 1)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        bootstrap_samples = [np.random.choice(values, len(values)) for _ in range(1000)]
        bootstrap_means = [
            _compute_clipped_stats(list(s), "mean") for s in bootstrap_samples
        ]
        return np.std(bootstrap_means)
    else:
        raise ValueError(f"Unknown {stat =}")


def _aggregate_get_clipped_mean(
    single_eval_results: list[SingleEvalResult],
) -> EvalResult:
    """
    Aggregate multiple SingleEvalResults into a single EvalResult for HealthBench.
    For each metric, returns the stats in _compute_clipped_stats.
    """
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    full_traces_list = []

    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)

        # Collect full_traces if available
        if (
            single_eval_result.example_level_metadata
            and "full_traces" in single_eval_result.example_level_metadata
        ):
            full_traces_list.append(
                single_eval_result.example_level_metadata["full_traces"]
            )
        else:
            full_traces_list.append(None)

    final_metrics = {}
    for name, values in name2values.items():
        for stat in ["mean", "n_samples", "bootstrap_std"]:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_clipped_stats(values, stat)

    # Include full_traces in metadata
    result_metadata = {
        "example_level_metadata": metadata,
        "full_traces": full_traces_list,
    }

    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata=result_metadata,
    )


class HealthBenchEval(Eval):
    def __init__(
        self,
        grader_model: (
            SamplerBase | None
        ) = None,  # Made optional for generation_only mode
        num_examples: int | None = None,
        n_repeats: int = 1,
        # If set, evaluate human completions or reference completions instead of model completions.
        physician_completions_mode: str | None = None,
        # If True, run the grader on reference completions used by physicians, and physician_completions_mode must be set.
        run_reference_completions: bool = False,
        n_threads: int = 120,
        subset_name: Literal["hard", "consensus", "all"] | None = None,
        generation_only: bool = False,  # New parameter for generation-only mode
        enable_checkpoint: bool = False,  # New parameter for checkpoint control
    ):
        if run_reference_completions:
            assert (
                physician_completions_mode is not None
            ), "physician_completions_mode must be provided if run_reference_completions is True"
            assert PHYSICIAN_COMPLETION_MODES[physician_completions_mode][
                "has_reference"
            ], "physician_completions_mode must have reference completions if run_reference_completions is True"

        if subset_name == "hard":
            input_path = INPUT_PATH_HARD
            input_local_path = (
                "./evaluation/health_bench_eval/data/hard_2025-05-08-21-00-10.jsonl"
            )
        elif subset_name == "consensus":
            input_path = INPUT_PATH_CONSENSUS
            input_local_path = "./evaluation/health_bench_eval/data/consensus_2025-05-09-20-00-46.jsonl"
        elif subset_name in (None, "all"):
            input_path = INPUT_PATH
            input_local_path = (
                "./evaluation/health_bench_eval/data/2025-05-07-06-14-12_oss_eval.jsonl"
            )
        else:
            raise ValueError(f"Invalid subset name: {subset_name}")

        if os.path.exists(input_local_path):
            # Load from local file if it exists
            with open(input_local_path, "r") as f:
                examples = [json.loads(line) for line in f]
        else:
            # Load from remote and save to local
            with urllib.request.urlopen(input_path) as response:
                examples = [json.loads(line.decode("utf-8")) for line in response]
            # Save to local file for future use
            os.makedirs(os.path.dirname(input_local_path), exist_ok=True)
            with open(input_local_path, "w") as f:
                for example in examples:
                    f.write(json.dumps(example) + "\n")

        for example in examples:
            example["rubrics"] = [RubricItem.from_dict(d) for d in example["rubrics"]]

        rng = random.Random(0)

        # physician completions mode
        self.physician_completions_mode = physician_completions_mode
        if self.physician_completions_mode is not None:
            assert (
                self.physician_completions_mode in PHYSICIAN_COMPLETION_MODES
            ), f"Invalid physician completions mode: {self.physician_completions_mode}; must be one of {PHYSICIAN_COMPLETION_MODES.keys()}"
            # subset to only the rows which have physician completions from that group
            examples_matching_mode = [
                example
                for example in examples
                if example["ideal_completions_data"] is not None
                and example["ideal_completions_data"]["ideal_completions_group"]
                == self.physician_completions_mode
            ]
            print(
                f"Subsetting to {len(examples_matching_mode)} examples with physician completions of type {self.physician_completions_mode} ({PHYSICIAN_COMPLETION_MODES[self.physician_completions_mode]['description']})"
            )

            examples = []
            if run_reference_completions:
                for example in examples_matching_mode:
                    for completion in example["ideal_completions_data"][
                        "ideal_completions_ref_completions"
                    ]:
                        new_example = copy.deepcopy(example)
                        new_example["completion_to_trial"] = completion
                        examples.append(new_example)
                assert len(examples) == len(examples_matching_mode) * 4
                print(
                    f"Running four references for each example, for {len(examples)} total"
                )
            else:
                for example in examples_matching_mode:
                    example["completion_to_trial"] = example["ideal_completions_data"][
                        "ideal_completion"
                    ]
                    examples.append(example)
                assert len(examples) == len(examples_matching_mode)

            if len(examples) == 0:
                raise ValueError(
                    f"No examples found matching mode {self.physician_completions_mode}"
                )

        if num_examples is not None and num_examples < len(examples):
            examples = rng.sample(
                examples,
                num_examples,
            )

        self.examples = examples * n_repeats
        self.n_threads = n_threads
        self.grader_model = grader_model
        self.generation_only = generation_only
        self.enable_checkpoint = enable_checkpoint

    def grade_sample(
        self,
        prompt: list[dict[str, str]],
        response_text: str,
        example_tags: list[str],
        rubric_items: list[RubricItem],
    ) -> tuple[dict, str, list[dict]]:
        # Skip grading in generation_only mode
        if self.generation_only or self.grader_model is None:
            # Return placeholder values for generation_only mode
            placeholder_metrics = {
                "overall_score": 0.0,
                "generation_only": True,
            }
            # Add placeholder scores for example-level tags
            example_tag_scores = {tag: 0.0 for tag in example_tags}
            placeholder_metrics.update(example_tag_scores)

            # Add placeholder scores for rubric-level tags
            rubric_tag_scores = {}
            for rubric_item in rubric_items:
                for tag in rubric_item.tags:
                    if tag not in rubric_tag_scores:
                        rubric_tag_scores[tag] = 0.0
            placeholder_metrics.update(rubric_tag_scores)

            # Create placeholder grading response list
            placeholder_grading_list = [
                {
                    "criteria_met": None,
                    "explanation": "Generation only - no evaluation performed",
                }
                for _ in rubric_items
            ]

            return placeholder_metrics, response_text, placeholder_grading_list

        # construct and grade the sample
        convo_with_response = prompt + [dict(content=response_text, role="assistant")]

        def grade_rubric_item(rubric_item: RubricItem) -> dict:
            convo_str = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in convo_with_response]
            )
            grader_prompt = GRADER_TEMPLATE.replace(
                "<<conversation>>", convo_str
            ).replace("<<rubric_item>>", str(rubric_item))
            messages: MessageList = [dict(content=grader_prompt, role="user")]
            while True:
                sampler_response = self.grader_model(messages)
                grading_response = sampler_response.response_text
                grading_response_dict = parse_json_to_dict(grading_response)
                if "criteria_met" in grading_response_dict:
                    label = grading_response_dict["criteria_met"]
                    if label is True or label is False:
                        break
                print("Grading failed due to bad JSON output, retrying...")
            return grading_response_dict

        grading_response_list = common.map_with_progress(
            grade_rubric_item,
            rubric_items,
            pbar=False,
        )

        # compute the overall score
        overall_score = calculate_score(rubric_items, grading_response_list)
        assert overall_score is not None
        metrics = {
            "overall_score": overall_score,
        }

        # compute scores for example-level tags)
        example_tag_scores = {tag: overall_score for tag in example_tags}
        assert len(example_tag_scores) == len(example_tags)  # No duplicates.
        metrics.update(example_tag_scores)

        # compute scores for rubric-level tags
        rubric_tag_items_grades = defaultdict(list)
        for rubric_item, grading_response in zip(rubric_items, grading_response_list):
            curr_item_tags = set()  # Ensure no duplicates in a rubric item.
            for tag in rubric_item.tags:
                rubric_tag_items_grades[tag].append((rubric_item, grading_response))
                assert tag not in curr_item_tags
                curr_item_tags.add(tag)

        rubric_tag_scores = {}
        for tag, items_grades in rubric_tag_items_grades.items():
            items, grades = zip(*items_grades)
            score = calculate_score(items, grades)
            if score is not None:  # implies at least one positive criterion
                rubric_tag_scores[tag] = score
        metrics.update(rubric_tag_scores)

        # construct the list of explanations and grades
        rubric_items_with_grades = []
        readable_explanation_list = []
        for rubric_item, grading_response in zip(rubric_items, grading_response_list):
            explanation = grading_response.get("explanation", "No explanation provided")
            criteria_met = grading_response["criteria_met"]
            readable_explanation = (
                f"[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}"
            )
            readable_explanation_list.append(readable_explanation)
            rubric_items_with_grades.append(
                {
                    **rubric_item.to_dict(),
                    "criteria_met": criteria_met,
                    "explanation": explanation,
                }
            )

        readable_explanation_list.sort(
            key=lambda x: x.startswith("[False]"), reverse=True
        )
        readable_explanation_str = "\n\n".join(readable_explanation_list)
        readable_explanation_str = f"\n\n{readable_explanation_str}"

        return metrics, readable_explanation_str, rubric_items_with_grades

    def generate(self, sampler: SamplerBase) -> list[dict]:
        """Generate responses for all examples and return raw generation data"""

        def generate_single(row: dict):
            prompt_messages = row["prompt"]

            if self.physician_completions_mode is not None:
                response_text = row["completion_to_trial"]
                response_usage = None
                actual_queried_prompt_messages = prompt_messages
                full_traces = None  # No traces available for physician completions
            else:
                sampler_response = sampler(prompt_messages)
                response_text = sampler_response.response_text
                response_dict = sampler_response.response_metadata
                actual_queried_prompt_messages = (
                    sampler_response.actual_queried_message_list
                )
                response_usage = response_dict.get("usage", None)
                full_traces = sampler_response.full_traces

            return {
                "row": row,
                "response_text": response_text,
                "actual_queried_prompt_messages": actual_queried_prompt_messages,
                "response_usage": response_usage,
                "full_traces": full_traces,
            }

        # Generate responses using threading
        if self.enable_checkpoint:
            # Include num_examples in checkpoint filename for proper cache separation
            num_examples_str = f"_n{len(self.examples)}"
            checkpoint_path = (
                f"eval_results/healthbench_generation{num_examples_str}.checkpoint.json"
            )
            generation_data = common.map_with_progress_checkpoint(
                generate_single,
                self.examples,
                checkpoint_path,
                num_threads=self.n_threads,
                checkpoint_interval=max(10, self.n_threads),
            )
        else:
            generation_data = common.map_with_progress(
                generate_single,
                self.examples,
                num_threads=self.n_threads,
                pbar=True,
            )
        return generation_data

    def evaluate(self, generation_data: list[dict]) -> list[SingleEvalResult]:
        """Evaluate generated responses and return SingleEvalResult objects"""
        if self.grader_model is None:
            raise ValueError(
                "grader_model is required for evaluation. Set eval_obj.grader_model before calling evaluate()."
            )

        # Temporarily disable generation_only mode for evaluation
        original_generation_only = self.generation_only
        self.generation_only = False

        def evaluate_single(gen_data: dict) -> SingleEvalResult:
            row = gen_data["row"]
            response_text = gen_data["response_text"]
            actual_queried_prompt_messages = gen_data["actual_queried_prompt_messages"]
            rubric_items = row["rubrics"]
            if isinstance(rubric_items[0], dict):
                rubric_items = [RubricItem.from_dict(d) for d in rubric_items]

            # Grade the response (generation_only is temporarily False)
            metrics, readable_explanation_str, rubric_items_with_grades = (
                self.grade_sample(
                    prompt=actual_queried_prompt_messages,
                    response_text=response_text,
                    rubric_items=rubric_items,
                    example_tags=row["example_tags"],
                )
            )

            score = metrics["overall_score"]

            return SingleEvalResult(
                score=score,
                metrics=metrics,
                pred_answer=response_text,
                id=row["id"],
            )

        results = common.map_with_progress(
            evaluate_single,
            generation_data,
            num_threads=self.n_threads,
            pbar=True,
        )

        return results

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """Legacy method for backward compatibility - orchestrates generate -> evaluate -> aggregate"""
        # Generate responses
        generation_data = self.generate(sampler)

        # Evaluate if not in generation_only mode
        if self.generation_only:
            # For generation_only, just return conversations and traces without evaluation results
            results = []
            for gen_data in generation_data:
                convo = gen_data["actual_queried_prompt_messages"] + [
                    dict(content=gen_data["response_text"], role="assistant")
                ]
                results.append(
                    SingleEvalResult(
                        html=None,
                        score=0.0,  # Placeholder
                        convo=convo,
                        metrics={"generation_only": True},
                        example_level_metadata={"full_traces": gen_data["full_traces"]},
                    )
                )
        else:
            # Full evaluation
            results = self.evaluate(generation_data)

        return _aggregate_get_clipped_mean(results)


def physician_completions_main(
    run_reference_completions: bool = False,
    num_examples: int | None = None,
    n_threads: int = 120,
    subset_name: Literal["hard", "consensus"] | None = None,
):
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")

    grading_sampler = ChatCompletionSampler(
        # model="gpt-4.1-2025-04-14",
        model="gpt-4-turbo",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        temperature=0,
    )
    dummy_sampler = SamplerBase()

    merge_metrics = []
    for pc_mode in PHYSICIAN_COMPLETION_MODES.keys():
        if (
            run_reference_completions
            and not PHYSICIAN_COMPLETION_MODES[pc_mode]["has_reference"]
        ):
            continue

        # run
        eval = HealthBenchEval(
            grader_model=grading_sampler,
            physician_completions_mode=pc_mode,
            run_reference_completions=run_reference_completions,
            num_examples=num_examples,
            n_threads=n_threads,
            subset_name=subset_name,
        )
        result = eval(dummy_sampler)

        # report
        parsable_mode = PHYSICIAN_COMPLETION_MODES[pc_mode]["short_name"]
        if run_reference_completions:
            file_stem = f"healthbench_{parsable_mode}_referencecompletions_{date_str}"
        else:
            file_stem = f"healthbench_{parsable_mode}_humanbaseline_{date_str}"
        report_filename = Path(f"/tmp/{file_stem}.html")
        report_filename.write_text(common.make_report(result))
        print(f"Report saved to {report_filename}")

        # metrics
        assert result.metrics is not None
        metrics = result.metrics
        result_filename = Path(f"/tmp/{file_stem}.json")
        result_filename.write_text(json.dumps(metrics))
        print(f"Results saved to {result_filename}")

        full_result_dict = {
            "score": result.score,
            "metrics": result.metrics,
            "htmls": result.htmls,
            "convos": result.convos,
            "metadata": result.metadata,
        }
        full_result_filename = Path(f"/tmp/{file_stem}_allresults.json")
        full_result_filename.write_text(json.dumps(full_result_dict, indent=2))
        print(f"All results saved to {full_result_filename}")

        # metrics df
        merge_metrics.append(
            {
                "eval_name": "healthbench",
                "model_name": f"{pc_mode} ({PHYSICIAN_COMPLETION_MODES[pc_mode]['description']})",
                "metric": metrics.get("overall_score", None),
            }
        )

    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


def gpt4_completions_main(
    num_examples: int | None = None,
    n_threads: int = 120,
    subset_name: Literal["hard", "consensus", "all"] | None = None,
    eval_save_dir: str = "health_bench_eval/eval_results",
):
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")

    grading_sampler = ChatCompletionSampler(
        model="gpt-4-turbo",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        temperature=0,
    )
    gpt4_sampler = ChatCompletionSampler(
        model="gpt-4-turbo",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=4096,
        temperature=0,
    )

    eval = HealthBenchEval(
        grader_model=grading_sampler,
        num_examples=num_examples,
        n_threads=n_threads,
        subset_name=subset_name,
    )
    result = eval(gpt4_sampler)

    # report
    report_filename = Path(
        f"{eval_save_dir}/healthbench_gpt4_completions_{date_str}.html"
    )
    report_filename.write_text(common.make_report(result))
    print(f"Report saved to {report_filename}")

    # metrics
    assert result.metrics is not None
    metrics = result.metrics
    result_filename = Path(
        f"{eval_save_dir}/healthbench_gpt4_completions_{date_str}.json"
    )
    result_filename.write_text(json.dumps(metrics))
    print(f"Results saved to {result_filename}")

    full_result_dict = {
        "score": result.score,
        "metrics": result.metrics,
        "htmls": result.htmls,
        "convos": result.convos,
        "metadata": result.metadata,
    }
    full_result_filename = Path(
        f"{eval_save_dir}/healthbench_gpt4_completions_{date_str}_allresults.json"
    )
    full_result_filename.write_text(json.dumps(full_result_dict, indent=2))
    print(f"All results saved to {full_result_filename}")


def cached_completions_main(
    num_examples: int | None = None,
    n_threads: int = 120,
    subset_name: Literal["hard", "consensus", "all"] | None = None,
    eval_save_dir: str = "health_bench_eval/eval_results",
    cached_completions_path: str = None,
):
    raise NotImplementedError(
        "Cached completions are not supported. Use workflow sampler instead."
    )
