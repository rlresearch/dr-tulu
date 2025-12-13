from __future__ import annotations

import argparse
import json
import os
import re
import string

# Ensure repo root on sys.path for module imports when running directly
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Add tqdm for progress bars
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Samplers
from samplers._types import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult
from samplers.sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)

# Try to import MCP sampler
try:
    from samplers.sampler.rl_rag_mcp_sampler import (
        RLRAGMCPSampler,
        create_mcp_sampler_with_websearch,
    )

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    RLRAGMCPSampler = None
    create_mcp_sampler_with_websearch = None

# Try to import Workflow sampler
try:
    from samplers.sampler.workflow_sampler import WorkflowSampler

    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False
    WorkflowSampler = None

# Datasets
from datasets import load_dataset
from dr_agent.dataset_utils.load_dataset import load_shortformqa_data

SUPPORTED_TASKS = {
    "2wiki_rlvr_no_prompt": "rulins/2wiki_rlvr_no_prompt",
    "tqa_rlvr_no_prompt": "rulins/tqa_rlvr_no_prompt",
    "nq_rlvr_no_prompt": "rulins/nq_rlvr_no_prompt",
    "hotpotqa_rlvr_no_prompt": "rulins/hotpotqa_rlvr_no_prompt",
    "gpqa_rlvr_no_prompt": "rl-rag/gpqa_diamond_rlvr_no_prompt",
    "2wiki": "akariasai/2wiki_rand1k",
    "tqa": "rulins/tqa_rlvr_no_prompt",
    "nq": "rulins/nq_rlvr_no_prompt",
    "hotpotqa": "rulins/hotpotqa_rlvr_no_prompt",
    "gpqa": "rl-rag/gpqa_diamond_rlvr_no_prompt",
    "bc_synthetic_v_2": "rl-rag/bc_synthetic_v_2",
    "webwalker": "rl-research/webwalker_test",
    "gaia": "rl-rag/gaia_dev",
    "bc_synthetic_depth_one_v2_verified": "rl-rag/verifiable_synthetic_depth_one_v2_verified",
    "bc_synthetic_varied_depth_o3_verified": "rl-rag/verifiable_synthetic_varied_depth_o3_verified",
    "hle": "rl-rag/hle_rlvr_no_prompt",
}


def get_ablation_sample_size(benchmark: str, subset_name: str = None) -> int:
    """Get the sample size for ablation studies (20% of full dataset)"""

    # Define full dataset sizes for each benchmark
    dataset_sizes = {
        "healthbench": {"hard": 183, "consensus": 183, "all": 366},  # hard + consensus
        "browsecomp": 1000,  # Approximate size
        "simpleqa": 4326,  # Full SimpleQA dataset size
        "researchqa": 100,  # Approximate size
    }

    if benchmark == "healthbench":
        if subset_name and subset_name in dataset_sizes[benchmark]:
            full_size = dataset_sizes[benchmark][subset_name]
        else:
            full_size = dataset_sizes[benchmark]["all"]
    else:
        full_size = dataset_sizes.get(benchmark, 100)  # Default fallback

    # Calculate 20% with minimum of 100 samples and maximum of 1000 samples
    ablation_size = min(max(100, int(full_size * 0.2)), 1000)

    return ablation_size


@dataclass
class ExampleRow:
    id: str
    question: str
    answers: List[str]


def normalize_text(s: str) -> str:
    """Relaxed normalization similar to SQuAD EM.
    - Lowercase
    - Remove punctuation
    - Remove articles
    - Collapse whitespace
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_exact_match(prediction: str, golds: Iterable[str]) -> bool:
    # hacky fix for format
    prediction = prediction.replace("Exact Answer: ", "").strip()

    pred_norm = normalize_text(prediction)

    for g in golds:
        if normalize_text(str(g)) == pred_norm:
            return True
    return False


def normalize_answer(s: str) -> str:
    """
    Normalize the answer by lowercasing, removing punctuation, articles,
    and extra whitespace.

    Based on:
    https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {"f1": 0, "precision": 0, "recall": 0}
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {"f1": f1, "precision": precision, "recall": recall}


def is_f1_match(prediction: str, golds: Iterable[str]) -> bool:
    f1 = max(f1_score(prediction, str(lab))["f1"] for lab in golds)
    return f1


class ShortFormQAEval(Eval):
    def __init__(
        self,
        task: str,
        split: str = "test",
        num_examples: Optional[int] = None,
        metric: str = "f1",  # "exact", "judge", or "generation_only"
        grader_model: Optional[SamplerBase] = None,
        enable_checkpoint: bool = False,
    ):
        # from datasets import load_dataset
        # ds = load_dataset(dataset_repo, split=split)

        # self.rows: List[ExampleRow] = []
        # for i, row in enumerate(ds):
        #     try:
        #         q, golds, rid = auto_extract(row)
        #         self.rows.append(ExampleRow(id=rid, question=q, gold_answers=golds))
        #     except Exception:
        #         continue
        # if num_examples:
        #     self.rows = self.rows[:num_examples]
        dataset_repo = SUPPORTED_TASKS[task]
        self.rows = load_shortformqa_data(dataset_repo, num_examples)

        print(f"Loaded {len(self.rows)} examples")
        self.metric = metric
        if "gpqa" in dataset_repo:
            self.metric = "exact"
            print(f"Using exact metric for gpqa")

        self.grader_model = grader_model
        self.enable_checkpoint = enable_checkpoint

    def _grade_with_judge(self, question: str, gold: List[str], response: str) -> bool:
        # Skip grading in generation_only mode or if no grader available
        if self.metric == "generation_only" or self.grader_model is None:
            return None

        assert self.grader_model is not None, "grader_model required for judge metric"
        # Ask judge if response matches any of the gold answers
        correct_answer = " / ".join(gold)
        grader_prompt = (
            "Judge whether the following [response] to [question] is correct based on the precise and unambiguous [correct_answer].\n\n"
            f"[question]: {question}\n\n[response]: {response}\n\n"
            "Your judgement must be:\n"
            "- extracted_final_answer: the final exact answer extracted from [response] (or None).\n"
            f"[correct_answer]: {correct_answer}\n\n"
            "reasoning: brief rationale focusing only on equivalence to [correct_answer].\n"
            "correct: yes or no (yes if equivalent or within trivial formatting differences).\n"
        )
        messages = [self.grader_model._pack_message(role="user", content=grader_prompt)]
        sampler_response = self.grader_model(messages)
        text = sampler_response.response_text or ""
        match = re.search(r"correct:\s*(yes|no)", text, flags=re.IGNORECASE)
        if not match:
            return False
        return match.group(1).lower() == "yes"

    def generate(self, sampler: SamplerBase) -> List[Dict]:
        """Generate responses for all examples and return raw generation data"""
        generation_data = []

        for row in tqdm(self.rows, desc="Generating responses"):
            messages = [
                sampler._pack_message(
                    role="user",
                    content=f"Answer concisely and exactly.\n\nQuestion: {row.question}",
                )
            ]
            sresp = sampler(messages)
            pred = sresp.response_text or ""
            # full_traces = sresp.full_traces

            generation_data.append(
                {
                    "row": row,
                    "response_text": pred,
                    "actual_queried_message_list": sresp.actual_queried_message_list,
                    # "full_traces": full_traces,
                    "question": row.question,
                    "gold_answers": row["answers"],
                }
            )

        return generation_data

    def evaluate(self, generation_data: List[Dict]) -> List[SingleEvalResult]:
        """Evaluate generated responses and return SingleEvalResult objects"""
        if self.metric == "judge" and self.grader_model is None:
            raise ValueError(
                "grader_model is required for judge metric evaluation. Set eval_obj.grader_model before calling evaluate()."
            )
        elif self.metric == "generation_only":
            raise ValueError(
                "Cannot evaluate with generation_only metric. Set eval_obj.metric to 'exact' or 'judge' before calling evaluate()."
            )

        results = []
        num_correct = 0

        for gen_data in generation_data:
            row = gen_data["row"]
            pred = gen_data["response_text"]

            # Always evaluate when in evaluate() method
            if self.metric == "exact":
                is_correct = is_exact_match(pred, row["answers"])
            elif self.metric == "f1":
                score = is_f1_match(pred, row["answers"])
                is_correct = score > 0.0
            elif self.metric == "judge":
                is_correct = self._grade_with_judge(
                    row["problem"], row["answers"], pred
                )
            else:
                raise ValueError(f"Unknown metric for evaluation: {self.metric}")

            if is_correct is not None and not self.metric == "f1":
                num_correct += 1 if is_correct else 0

            if self.metric == "f1":
                score = score
            else:
                score = float(is_correct) if is_correct is not None else 0.0
            metrics = {"is_correct": is_correct}

            results.append(
                SingleEvalResult(
                    score=score,
                    metrics=metrics,
                    gt_answer=row["answers"],
                    pred_answer=pred,
                    id=row["id"],
                )
            )

        return results

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """Legacy method for backward compatibility - orchestrates generate -> evaluate -> aggregate"""
        # Generate responses
        generation_data = self.generate(sampler)

        # Handle generation_only mode vs evaluation mode
        if self.metric == "generation_only":
            # For generation_only, just return conversations and traces without evaluation results
            results = []
            for gen_data in generation_data:
                convo = gen_data["actual_queried_message_list"] + [
                    dict(role="assistant", content=gen_data["response_text"])
                ]
                # Append ground truth for easy comparison in logs
                convo.append(
                    dict(role="system", content=f"Gold answers: {gen_data['answers']}")
                )

                results.append(
                    SingleEvalResult(
                        score=0.0,  # Placeholder
                        metrics={"generation_only": True},
                        html=None,
                        convo=convo,
                        example_level_metadata={
                            "id": gen_data["row"].id,
                            "question": gen_data["question"],
                            "gold_answers": gen_data["answers"],
                            "prediction": gen_data["response_text"],
                            # "full_traces": gen_data["full_traces"],
                        },
                    )
                )
        else:
            # Full evaluation
            results = self.evaluate(generation_data)

        # Use common aggregation to properly handle full_traces
        from samplers import common

        return common.aggregate_results(results)
