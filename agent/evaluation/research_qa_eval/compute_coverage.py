import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import openai
from retry import retry
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom exception for batch processing failures
class BatchProcessingError(Exception):
    pass


@dataclass
class RubricItem:
    rubric_item: str
    type: str
    citation_metadata: Optional[Dict[str, str]] = None


@dataclass
class ResearchQAItem:
    id: str
    general_domain: str
    subdomain: str
    field: str
    query: str
    date: str
    rubric: List[RubricItem]


def load_researchqa_data(json_path: str) -> List[ResearchQAItem]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for item in data:
        rubric = [RubricItem(**r) for r in item["rubric"]]
        items.append(
            ResearchQAItem(
                id=item["id"],
                general_domain=item["general_domain"],
                subdomain=item["subdomain"],
                field=item["field"],
                query=item["query"],
                date=item["date"],
                rubric=rubric,
            )
        )
    return items


def call_gpt(prompt: str, model: str = "gpt-4") -> str:
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response.choices[0].message.content.strip()


def build_prompt(response: str, questions: List[str]) -> str:
    prompt = (
        "Please judge the following questions based on the response below.\n"
        "For each question, select one of the following ratings to indicate the extent to which the response addresses the question:\n"
        "Not at all, Barely, Moderately, Mostly, Completely\n\n"
        "Definitions:\n"
        "- Not at all: *totally uninferable*\n"
        "- Barely: *unmentioned but inferrable*\n"
        "- Moderately: *mentioned but misses important details*\n"
        "- Mostly: *mentioned but misses some details*\n"
        "- Completely: *mentioned with sufficient details*\n\n"
        "Only output one of the five phrases for each question, separated by newlines, and nothing else.\n\n"
        f"Response: {response}\n"
        f"Questions:\n" + "\n".join(questions) + "\n\nOutput:"
    )
    return prompt


def convert_text_to_score(text):
    text_to_score = {
        "Not at all": 1,
        "Barely": 2,
        "Moderately": 3,
        "Mostly": 4,
        "Completely": 5,
    }
    score = text_to_score[text]
    return score


def normalize_5scale(x):
    return (x - 1) / 4


def compute_coverage_score(rubric_judges):
    """
    rubric_judges: list of dicts, each with a 'score' key (text label)
    Returns: average normalized coverage score (float) in [0, 1]
    """
    numeric_scores = [convert_text_to_score(j["score"]) for j in rubric_judges]
    normalized_scores = [normalize_5scale(x) for x in numeric_scores]
    return sum(normalized_scores) / len(normalized_scores)


@retry(tries=3, delay=1, backoff=2, exceptions=BatchProcessingError)
def process_batch_with_retry(
    prompt: str, batch: List[str], model: str, item_id: str
) -> List[str]:
    """
    Process a batch of rubric items with retry logic.

    Args:
        prompt: The prompt to send to GPT
        batch: List of rubric items in the batch
        model: The model name for logging purposes
        item_id: The item ID for logging purposes

    Returns:
        List of scores for the batch
    """
    gpt_output = call_gpt(prompt, model=model)
    scores = gpt_output.split("\n")

    # Filter out empty lines
    scores = [score.strip() for score in scores if score.strip()]

    if len(scores) == len(batch):
        return scores
    else:
        logger.warning(
            f"Expected {len(batch)} results for item {item_id}, got {len(scores)}. Retrying..."
        )
        logger.warning(f"Questions: {batch}")
        raise BatchProcessingError(f"Expected {len(batch)} results, got {len(scores)}")


def compute_coverage(
    data_path: str,
    response_map: Dict[str, Dict[str, str]],
    output_path: Optional[str] = None,
    batch_size: int = 8,
    model: str = "gpt-4.1-mini",
):
    # Load questions
    items = load_researchqa_data(data_path)

    results = {}
    coverages = []

    original_items = items
    items = [item for item in items if item.id in response_map]
    print(
        f"Computing coverage for {len(items)} items out of {len(original_items)} total items."
    )
    for item in tqdm(items, desc="Computing rubric coverage"):
        item_id = item.id
        answer = response_map[item_id]["answer"]
        rubric_items = [r.rubric_item for r in item.rubric]
        rubric_judges = []
        item_processing_failed = False

        for i in range(0, len(rubric_items), batch_size):
            batch = rubric_items[i : i + batch_size]
            prompt = build_prompt(answer, batch)

            try:
                # Process batch with retry
                scores = process_batch_with_retry(prompt, batch, model, item_id)

                # Add results to rubric_judges
                for rubric, score in zip(batch, scores):
                    rubric_judges.append(
                        {
                            "rubric": rubric,
                            "score": score,
                        }
                    )

            except Exception as e:
                logger.error(
                    f"Failed to get correct number of results for batch after all retries. Skipping item {item_id}. Error: {e}"
                )
                item_processing_failed = True
                break

        # Skip this item if any batch failed
        if item_processing_failed:
            continue

        results[item_id] = rubric_judges
        # Compute and print coverage for this item
        item_coverage = compute_coverage_score(rubric_judges)
        coverages.append(item_coverage)
        logger.info(f"Coverage for {item_id}: {item_coverage:.3f}")

    if output_path:
        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved rubric coverage results to {output_path}")

        # Print overall average coverage
        if coverages:
            avg_coverage = sum(coverages) / len(coverages)
            logger.info(f"Average rubric coverage: {avg_coverage:.3f}")
        else:
            logger.info("No coverage scores computed.")

    return results, coverages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute rubric coverage using GPT")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to ResearchQA JSON file"
    )
    parser.add_argument(
        "--responses",
        type=str,
        required=True,
        help="Path to response map JSON file (id -> response dict)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save rubric coverage results"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for rubric items"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4.1-mini", help="OpenAI model name"
    )
    args = parser.parse_args()
    compute_coverage(
        args.data, args.responses, args.output, args.batch_size, args.model
    )
