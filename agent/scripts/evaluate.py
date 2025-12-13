#!/usr/bin/env python3
"""
Simple evaluation script for generated JSONL data files.
Auto-detects task type and runs appropriate evaluation.

Usage:
    python evaluate_results.py <task> <path/to/results.jsonl>

Examples:
    python evaluate_results.py simpleqa eval_output/baselines-20250828/sft_search_generate/simpleqa-ablation.jsonl
    python evaluate_results.py browsecomp eval_output/baselines-20250828/sft_search_generate/browsecomp-ablation.jsonl
    python evaluate_results.py healthbench eval_output/baselines-20250828/sft_search_generate/healthbench-ablation.jsonl
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "evaluation"))

import argparse
import json
import os

from dotenv import load_dotenv
from samplers import common

from evaluation.browse_comp_eval.browsecomp_eval import BrowseCompEval
from evaluation.health_bench_eval.healthbench_eval import HealthBenchEval
from evaluation.research_qa_eval.compute_coverage import compute_coverage
from evaluation.research_qa_eval.researchqa_eval import ResearchQAEval
from evaluation.genetic_diseases_eval.genetic_diseases_eval import GeneticDiseasesEval
from evaluation.research_qa_eval.researchqa_loader import download_researchqa_dataset
from evaluation.samplers.sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)
from evaluation.short_form_qa_eval.short_form_eval import ShortFormQAEval
from evaluation.simple_qa_eval.simpleqa_eval import SimpleQAEval

load_dotenv()


def load_jsonl(file_path):
    """Load a JSON Lines file and return a list of dictionaries"""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def convert_to_evaluate_format(original_examples, add_traces: bool = False):
    """Convert original examples to evaluation format"""
    converted_examples = []
    for ele in original_examples:
        example = {
            "id": ele["example_id"],
            "row": ele["original_data"],
            "response_text": ele["final_response"],
            "actual_queried_prompt_messages": [
                {"content": ele["problem"], "role": "user"}
            ],
        }
        if add_traces:
            example["full_traces"] = ele["full_traces"]
        converted_examples.append(example)

    return converted_examples


def evaluate_researchqa(
    file_path,
    save_path,
    grader_model="gpt-4.1-mini",
):

    # Save results to same folder
    input_dir = Path(file_path).parent
    input_name = Path(file_path).stem  # filename without extension
    if save_path is None:
        results_filename = f"{input_name}_eval_results.json"
        results_path = input_dir / results_filename
    else:
        results_path = save_path

    original_examples = load_jsonl(file_path)
    response_map = {
        ele["original_data"]["orig_id"]: {
            "answer": ele["final_response"],
        }
        for ele in original_examples
    }

    download_researchqa_dataset(
        split="test.json", output_dir="evaluation/research_qa_eval/data"
    )

    results, coverages = compute_coverage(
        data_path="evaluation/research_qa_eval/data/test.json",
        response_map=response_map,
        output_path=None,
        batch_size=10,
        model=grader_model,
    )

    results_data = {
        "task": "researchqa",
        "run_mode": "evaluation",
        "score": sum(coverages) / len(coverages),
        "metrics": {
            "coverage": sum(coverages) / len(coverages),
        },
        "metadata": {},
        "num_examples": len(original_examples),
    }
    if results:
        results_data["per_example_results"] = results

    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)


def evaluate_genetic_disease_qa(
    file_path: str,
    save_path: str,
    run_mode: str,
    grader_model: str ="gpt-4.1-2025-04-14",
    debug: bool = False,
):
    task = "genetic_qa"
    converted_examples = load_jsonl(file_path)
    if "original_data" in converted_examples[0].keys():
        converted_examples = convert_to_evaluate_format(converted_examples, add_traces=True)
    if debug:
        converted_examples = converted_examples[:1]
    print(f"Loaded {len(converted_examples)} examples")

    grader_sampler = ChatCompletionSampler(
        model=grader_model,
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=1000,
        temperature=0,
    )
    # must specify run_mode, as citation format differs across models
    eval_class = GeneticDiseasesEval(grader_model=grader_sampler, run_mode=run_mode)

    # Setup save path
    input_dir = Path(file_path).parent
    input_name = Path(file_path).stem  # filename without extension
    results_path = save_path if save_path is not None else input_dir / f"{input_name}_eval_results.json"
    
    # Run evaluation
    eval_results = eval_class.evaluate(converted_examples)
    final_result = common.aggregate_results(eval_results)

    save_evaluation_results(
        str(results_path), task, run_mode, final_result, converted_examples
    )

    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Task: {task}")
    print(f"Examples: {len(converted_examples)}")
    print(f"Score: {final_result.score:.3f}")
    print(f"Results saved to: {results_path}")

    return final_result


def save_evaluation_results(
    results_path: str,
    task_name: str,
    run_mode: str,
    final_result,
    generation_data: list,
):
    """Save evaluation results to results file"""
    results_data = {
        "task": task_name,
        "run_mode": run_mode,
        "score": final_result.score,
        "metrics": final_result.metrics,
        "metadata": final_result.metadata,
        "num_examples": len(generation_data),
    }
    if final_result.per_example_results:
        results_data["per_example_results"] = final_result.per_example_results

    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"[SAVED] Results saved to {results_path}")


def run_evaluation(
    file_path,
    save_path,
    task,
    task_type,
    grader_model="gpt-4o-mini",
    debug=False,
):
    """Run evaluation on the specified JSONL file"""

    print(f"Task: {task}")
    print(f"Task type: {task_type}")

    # Load data
    print(f"Loading data from: {file_path}")
    original_examples = load_jsonl(file_path)

    if debug:
        original_examples = original_examples[:5]

    converted_examples = convert_to_evaluate_format(original_examples)
    print(f"Loaded {len(converted_examples)} examples")

    # Setup grader
    grader_sampler = ChatCompletionSampler(
        model=grader_model,
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=1000,
        temperature=0,
    )

    # Run appropriate evaluation
    print(f"Running {task} evaluation...")

    if task == "simpleqa":
        eval_class = SimpleQAEval(grader_model=grader_sampler)
    elif task == "browsecomp":
        eval_class = BrowseCompEval(grader_model=grader_sampler)
    elif task == "healthbench":
        eval_class = HealthBenchEval(grader_model=grader_sampler)
    elif task in ["2wiki", "webwalker"]:
        eval_class = ShortFormQAEval(
            task=task, grader_model=grader_sampler, metric="judge"
        )
    else:
        raise ValueError(f"Unsupported task type: {task}")

    # Run evaluation
    eval_results = eval_class.evaluate(converted_examples)
    final_result = common.aggregate_results(eval_results)

    # Save results to same folder
    input_dir = Path(file_path).parent
    input_name = Path(file_path).stem  # filename without extension
    if save_path is None:
        results_filename = f"{input_name}_eval_results.json"
        results_path = input_dir / results_filename
    else:
        results_path = save_path

    save_evaluation_results(
        str(results_path), task, "evaluation", final_result, converted_examples
    )

    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Task: {task}")
    print(f"Examples: {len(converted_examples)}")
    print(f"Score: {final_result.score:.3f}")
    print(f"Results saved to: {results_path}")

    return final_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate JSONL results file")
    parser.add_argument(
        "task",
        help="Task type to evaluate",
    )
    parser.add_argument("file_path", help="Path to the JSONL results file")
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--task_type",
        default=None,
        help="Task type to evaluate",
    )
    parser.add_argument(
        "--grader-model",
        default="gpt-4.1-2025-04-14",
        help="Model to use for grading (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--run_mode",
        default="auto_reason_search",
        help="Name of run mode used to generate response (specific to genetic diseases qa)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)

    if not args.file_path.endswith(".jsonl"):
        print(f"Error: File must be a JSONL file: {args.file_path}")
        sys.exit(1)

    if args.task == "researchqa":
        evaluate_researchqa(
            args.file_path,
            args.save_path,
            grader_model="gpt-4.1-mini-2025-04-14",  # hardcode for researchqa
        )
    elif args.task == "genetic_diseases_qa":
        # Must pass in the run_mode here so we know how to parse the
        # citation sources for evaluation
        evaluate_genetic_disease_qa(
            args.file_path,
            args.save_path,
            args.run_mode,
            grader_model=args.grader_model,
            debug=args.debug,
        )
    else:
        run_evaluation(
            args.file_path,
            args.save_path,
            args.task,
            args.task_type,
            args.grader_model,
            args.debug,
        )


if __name__ == "__main__":
    main()
