import hashlib
import random
from typing import Dict, List, Optional

from datasets import load_dataset

from .data_types import DatasetConfig

SUPPORTED_TASKS = {
    "genetic_qa": "parkmoll/genetic-variants-qa",
    "deep_scholar_bench": "xinranz3/deepscholar_bench_fixed",
    "sqav2": "allenai/asta-bench",
}


def get_ablation_sample_size(benchmark: str) -> int:
    """Get the sample size for ablation studies (20% of full dataset)"""
    dataset_sizes = {
        "deep_scholar_bench": 63,
        "sqav2": 100,
        "genetic_qa": 100,
    }

    full_size = dataset_sizes.get(benchmark, 100)
    ablation_size = min(max(100, int(full_size * 0.2)), 500)
    return ablation_size


def load_eval_dataset(config: DatasetConfig) -> List[Dict]:
    """
    Load evaluation dataset using configuration object.

    Args:
        config: DatasetConfig specifying which dataset to load

    Returns:
        List of dataset examples
    """
    num_examples = config.get("num_examples")
    if num_examples == "ablation":
        num_examples = get_ablation_sample_size(config["name"])
        shuffle = False
    elif num_examples == "final_run":
        num_examples = 1000
        shuffle = True
    elif num_examples == "final_run_100":
        num_examples = 100
        shuffle = True
    else:
        shuffle = False

    if isinstance(num_examples, str):
        raise ValueError(
            "num_examples must be an integer or 'ablation', 'final_run', or 'final_run_100'"
        )

    if config["name"] == "deep_scholar_bench":
        return load_deep_scholar_bench_data(num_examples)
    elif config["name"] == "sqav2":
        return load_sqav2_data(num_examples, shuffle)
    elif config["name"] == "genetic_qa":
        return load_genetic_qa_data(num_examples, shuffle)
    else:
        raise ValueError(
            f"Unsupported dataset: {config['name']}. Supported datasets: {list(SUPPORTED_TASKS.keys())}"
        )


def load_deep_scholar_bench_data(num_examples: Optional[int] = None) -> List[Dict]:
    """Load Deep Scholar Bench dataset data."""
    raw_data = load_dataset("xinranz3/deepscholar_bench_fixed", "default")

    examples = []
    for sample in raw_data["train"]:
        examples.append(
            {
                "id": sample["qid"],
                "problem": sample["query"],
                "additional_instructions": "",
            }
        )

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_sqav2_data(
    num_examples: Optional[int] = None, shuffle: bool = False
) -> List[Dict]:
    """Load SQA v2 dataset data."""
    data = load_dataset(
        "allenai/asta-bench",
        data_files="tasks/sqa/rubrics_v2_recomputed.json",
        split="train",
    )

    examples = []
    for sample in data:
        examples.append(
            {
                "id": sample["case_id"],
                "problem": sample["question"],
                "additional_instructions": "Please write a well structured, data-driven report on the given research question, and add citations when needed.",
            }
        )

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_genetic_qa_data(
    num_examples: Optional[int] = None, shuffle: bool = False
) -> List[Dict]:
    """Load Genetic Variants QA dataset data."""
    dataset_repo = SUPPORTED_TASKS["genetic_qa"]

    question_types_data = load_dataset(
        dataset_repo, data_files="question_types.json", split="train"
    )
    rare_variants_data = load_dataset(
        dataset_repo, data_files="rare_variants_qa.json", split="train"
    )

    question_types = question_types_data[0]
    rare_variants = rare_variants_data

    examples = []
    for example in rare_variants:
        question_type = example["question_type"]
        variant = example["variant"]
        template = question_types[question_type]["template"]
        problem = template.replace("{variant}", variant)

        examples.append(
            {
                "id": hashlib.md5(problem.encode()).hexdigest(),
                "problem": problem,
                "additional_instructions": "",
            }
        )

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples
