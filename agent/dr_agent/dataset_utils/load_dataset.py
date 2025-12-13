import base64
import hashlib
import json
import random
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import datasets
import pandas as pd
from huggingface_hub import hf_hub_download

from .data_types import DatasetConfig

SUPPORTED_TASKS = {
    "genetic_diseases_qa": "rl-research/genetic_diseases_qa",
    "deep_scholar_bench": "xinranz3/deepscholar_bench_fixed",
    "deep_research_bench": "rl-research/deep_research_bench_eval",
    "sqav2": "allenai/asta-bench",
    "researchqa": "realliyifei/ResearchQA",
    "2wiki": "akariasai/2wiki_rand1k",
    "webwalker": "rl-research/webwalker_test",
}

DATASET_URLS = {
    "browsecomp": "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv",
    "simpleqa": "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
    "healthbench_all": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "healthbench_hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
    "healthbench_consensus": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl",
}


def get_cache_dir() -> Path:
    """Get the cache directory for downloaded datasets."""
    cache_dir = Path.home() / ".cache" / "dr_agent" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, cache_name: str) -> Path:
    """Download file from URL to cache directory if not already cached."""
    cache_path = get_cache_dir() / cache_name
    if not cache_path.exists():
        urllib.request.urlretrieve(url, cache_path)
    return cache_path


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def get_ablation_sample_size(benchmark: str, subset_name: str = None) -> int:
    """Get the sample size for ablation studies (20% of full dataset)"""
    dataset_sizes = {
        "healthbench": {"hard": 183, "consensus": 183, "all": 366},
        "browsecomp": 1000,
        "simpleqa": 1000,
        "researchqa": 100,
        "deep_scholar_bench": 63,
        "deep_research_bench": 100,
        "sqav2": 100,
        "genetic_diseases_qa": 47,
        "2wiki": 100,
        "webwalker": 680,
    }

    if benchmark == "healthbench":
        if subset_name and subset_name in dataset_sizes[benchmark]:
            full_size = dataset_sizes[benchmark][subset_name]
        else:
            full_size = dataset_sizes[benchmark]["all"]
    else:
        full_size = dataset_sizes.get(benchmark, 100)

    ablation_size = min(max(100, int(full_size * 0.2)), 500)
    return ablation_size


def load_dataset(config: DatasetConfig) -> List[Dict]:
    """
    Load dataset using configuration object.

    Args:
        config: DatasetConfig specifying which dataset to load

    Returns:
        List of dataset examples
    """
    num_examples = config.get("num_examples")
    local_path = config.get("local_path")

    if num_examples == "ablation":
        num_examples = get_ablation_sample_size(config["name"], config.get("subset"))
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

    if config["name"] == "browsecomp":
        return load_browsecomp_data(num_examples, shuffle, local_path)
    elif config["name"] == "simpleqa":
        return load_simpleqa_data(num_examples, shuffle, local_path)
    elif config["name"] == "healthbench":
        subset = config.get("subset", "all") or "all"
        return load_healthbench_data(subset, num_examples, shuffle, local_path)
    elif config["name"] == "researchqa":
        return load_researchqa_data(num_examples, shuffle)
    elif config["name"] == "deep_scholar_bench":
        return load_deep_scholar_bench_data(num_examples)
    elif config["name"] == "deep_research_bench":
        return load_deep_research_bench_data(num_examples, shuffle)
    elif config["name"] == "sqav2":
        return load_sqav2_data(num_examples, shuffle)
    elif config["name"] == "genetic_diseases_qa":
        return load_genetic_diseases_qa_data(num_examples, shuffle)
    elif config["name"] in ["2wiki", "webwalker"]:
        dataset_repo = SUPPORTED_TASKS[config["name"]]
        return load_shortformqa_data(dataset_repo, num_examples, shuffle)
    else:
        raise ValueError(
            f"Unsupported dataset: {config['name']}. Supported datasets: {list(SUPPORTED_TASKS.keys())}, browsecomp, simpleqa, healthbench, researchqa, deep_research_bench"
        )


def load_browsecomp_data(
    num_examples: Optional[int] = None,
    shuffle: bool = False,
    local_path: Optional[str] = None,
) -> List[Dict]:
    """
    Load BrowseComp dataset data with decrypted problem and answer fields.

    Args:
        num_examples: Limit to first N examples (optional)
        shuffle: Whether to shuffle the examples
        local_path: Optional local path to dataset file

    Returns:
        List of BrowseComp examples with decrypted content
    """
    if local_path and Path(local_path).exists():
        df = pd.read_csv(local_path)
    else:
        cache_path = download_file(
            DATASET_URLS["browsecomp"], "browse_comp_test_set.csv"
        )
        df = pd.read_csv(cache_path)

    df["problem"] = df.apply(lambda row: decrypt(row["problem"], row["canary"]), axis=1)
    df["answer"] = df.apply(lambda row: decrypt(row["answer"], row["canary"]), axis=1)
    df["id"] = df["problem"].apply(
        lambda problem: hashlib.md5(problem.encode()).hexdigest()
    )
    df["additional_instructions"] = (
        """
Your final response should be in the following format:
Explanation: <your explanation for your final answer>
Exact Answer: <your succinct, final answer>
Confidence: <your confidence score between 0% and 100% for your answer>
""".strip()
    )

    examples = df.to_dict("records")

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_simpleqa_data(
    num_examples: Optional[int] = None,
    shuffle: bool = False,
    local_path: Optional[str] = None,
) -> List[Dict]:
    """
    Load SimpleQA dataset data.

    Args:
        num_examples: Limit to first N examples (optional)
        shuffle: Whether to shuffle the examples
        local_path: Optional local path to dataset file

    Returns:
        List of SimpleQA examples
    """
    if local_path and Path(local_path).exists():
        df = pd.read_csv(local_path)
    else:
        cache_path = download_file(DATASET_URLS["simpleqa"], "simple_qa_test_set.csv")
        df = pd.read_csv(cache_path)

    df["id"] = df["problem"].apply(
        lambda problem: hashlib.md5(problem.encode()).hexdigest()
    )
    df["additional_instructions"] = (
        """
Your final response should be in the following format:
Exact Answer: <your succinct, final answer>
""".strip()
    )

    examples = df.to_dict("records")

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_healthbench_data(
    subset: str = "all",
    num_examples: Optional[int] = None,
    shuffle: bool = False,
    local_path: Optional[str] = None,
) -> List[Dict]:
    """
    Load HealthBench dataset data.

    Args:
        subset: "all", "hard", or "consensus"
        num_examples: Limit to first N examples (optional)
        shuffle: Whether to shuffle the examples
        local_path: Optional local path to dataset file

    Returns:
        List of HealthBench examples
    """

    def format_conversation(conversation):
        role_mapping = {"user": "Patient", "assistant": "Physician"}
        return "\n\n".join(
            [f"{role_mapping[ele['role']]}: {ele['content']}" for ele in conversation]
        )

    def stringify_example_prompt(prompt):
        if len(prompt) == 1:
            return f"Can you answer the question from a patient about a medical condition or concern they have: {prompt[0]['content']}"
        elif len(prompt) > 1:
            return f"Here is a conversation between a patient and a physician. The patient is asking a question about a medical condition or concern they have, and in the conversation it should contain necessary background information about the patient:\n\n{format_conversation(prompt[:-1])}\n\nCan you search for needed information and answer the patient's question: {prompt[-1]['content']}"
        else:
            raise ValueError(
                "Data error: there should be at least one element in the prompt."
            )

    examples = []

    if local_path and Path(local_path).exists():
        with open(local_path, "r") as f:
            examples = [json.loads(line.strip()) for line in f if line.strip()]
    else:
        url_key = f"healthbench_{subset}"
        cache_name = f"healthbench_{subset}.jsonl"
        cache_path = download_file(DATASET_URLS[url_key], cache_name)
        with open(cache_path, "r") as f:
            examples = [json.loads(line.strip()) for line in f if line.strip()]

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    for example in examples:
        example["id"] = example["prompt_id"]
        example["problem"] = stringify_example_prompt(example["prompt"])

    return examples


def load_deep_scholar_bench_data(num_examples: Optional[int] = None) -> List[Dict]:
    """Load Deep Scholar Bench dataset data."""
    raw_data = datasets.load_dataset("xinranz3/deepscholar_bench_fixed", "default")

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
    data = datasets.load_dataset(
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


def load_genetic_diseases_qa_data(
    num_examples: Optional[int] = None, shuffle: bool = False
) -> List[Dict]:
    """Load Genetic Variants QA dataset data."""
    dataset_repo = SUPPORTED_TASKS["genetic_diseases_qa"]

    question_types_data = datasets.load_dataset(
        dataset_repo, data_files="question_types.json", split="train"
    )
    rare_variants_data = datasets.load_dataset(
        dataset_repo, data_files="genetic_diseases_qa.json", split="train"
    )

    question_types = question_types_data[0]
    rare_variants = rare_variants_data

    examples = []
    for example in rare_variants:
        question_type = example["question_type"]
        variant = example["variant"]
        template = question_types[question_type]["template"]
        problem = template.replace("{variant}", variant)

        general_rubrics = question_types[question_type]["rubrics"]
        instance_rubrics = example["rubrics"]

        examples.append(
            {
                "id": hashlib.md5(problem.encode()).hexdigest(),
                "problem": problem,
                "additional_instructions": "",
                "rubrics": general_rubrics + instance_rubrics
            }
        )

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_shortformqa_data(
    dataset_repo: str, num_examples: Optional[int] = None, shuffle: bool = False
) -> List[Dict]:
    """
    Load Short-form QA dataset data.

    Args:
        dataset_repo: HuggingFace dataset repository name
        num_examples: Limit to first N examples (optional)
        shuffle: Whether to shuffle the examples

    Returns:
        List of Short-form QA examples
    """
    dataset = datasets.load_dataset(dataset_repo, split="test")
    examples = []
    for example in dataset:
        example["problem"] = example["messages"][-1]["content"]
        example["id"] = hashlib.md5(example["problem"].encode()).hexdigest()
        example["answers"] = (
            json.loads(example["ground_truth"])
            if example["ground_truth"][0] == "["
            else [example["ground_truth"]]
        )
        example["additional_instructions"] = (
            """
Your final response should be in the following format without any other text:
Exact Answer: <your succinct, final answer>
""".strip()
        )
        examples.append(example)

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_deep_research_bench_data(
    num_examples: Optional[int] = None, shuffle: bool = False
) -> List[Dict]:
    """
    Load Deep Research Bench dataset data.

    Args:
        num_examples: Limit to first N examples (optional)
        shuffle: Whether to shuffle the examples

    Returns:
        List of Deep Research Bench examples
    """
    data = datasets.load_dataset("rl-research/deep_research_bench_eval", split="test")

    examples = []
    for sample in data:
        examples.append(
            {
                "id": sample["id"],
                "problem": sample["prompt"],
                "additional_instructions": "Please write a well structured, data-driven report on the given research question, and add citations when needed.",
            }
        )

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples


def load_researchqa_data(
    num_examples: Optional[int] = None,
    shuffle: bool = False,
) -> List[Dict]:
    """
    Load ResearchQA dataset data.
    Always loads from the official subset if the IDs file is available.

    Args:
        num_examples: Limit to first N examples (optional)
        shuffle: Whether to shuffle the examples

    Returns:
        List of ResearchQA examples
    """
    dataset_repo = SUPPORTED_TASKS["researchqa"]
    data = datasets.load_dataset(
        dataset_repo, split="test", revision="87cdd81df0c5ea96de293859233e8e64dac3d168"
    )

    # Try to load official subset IDs if available
    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = hf_hub_download(
            repo_id="rl-research/researchqa_official_subset_ids",
            filename="researchqa_official_subset_ids.json",
            repo_type="dataset",
            cache_dir=temp_dir,
        )

        # Load the JSON file directly (it's a list of strings)
        with open(json_path, "r") as f:
            official_ids = json.load(f)

    data = data.filter(lambda x: x["id"] in official_ids)

    examples = []
    for sample in data:
        examples.append(
            {
                "id": hashlib.md5(sample["query"].encode()).hexdigest(),
                "orig_id": sample["id"],
                "problem": sample["query"],
                "additional_instructions": "Answer the question completely and precisely in around 240-260 words. You need to support every statement in the answer with in-line citations to passages given in the context. Don't enumerate the facts. You should provide an answer in one-to-three paragraphs.",
            }
        )

    if shuffle:
        random.seed(42)
        random.shuffle(examples)

    if num_examples:
        examples = examples[:num_examples]

    return examples
