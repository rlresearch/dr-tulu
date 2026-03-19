import re
from typing import Dict, List

OLYMPIAD_JUDGE_PROMPT = """
You are grading an attempted answer to a science olympiad problem. You will be given the
problem, attempted answer, and reference answer. Evaluate the solution against the provided reference solution, ensuring it is complete and matches the reference solution. Pay close attention to detail and grade it strictly, but fairly.

The reference answer is either a single number or expression in latex formatting, a chemical formula, a compound name, or a phrase referring to a specific name, entity, or method.

Mark the attempted answer as correct if it fully matches the reference answer or is otherwise equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal place rounding of the reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a compound/formula, equivalent when accounting for units, etc.). Mark it as incorrect if it is not equivalent to the reference answer.

***
The problem: {problem}
***
The reference answer: {reference answer}
***
The attempted answer: {answer}
***

First, think step-by-step about whether the attempted answer matches the reference answer. If the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your response, with no other text or formatting. If it is incorrect, write "VERDICT: INCORRECT".
""".strip()

RESEARCH_JUDGE_PROMPT = """
You are grading a science exam.

You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to 10 points.

Evaluate the attemped answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric).

***
The problem: {problem}
***
The rubric: {rubric}
***
The attempted answer: {answer}
***

First, think step-by-step about each rubric item. Explain your reasoning for each rubric item. Then, tally the points up and write VERDICT: <total_points> in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8.
""".strip()

from evaluation.samplers import common
from evaluation.samplers._types import Eval, SamplerBase, SingleEvalResult

def _extract_verdict_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""

class FrontierScienceOlympiadEval(Eval):
    def __init__(self, grader_model: SamplerBase | None = None):
        self.grader_model = grader_model

    @staticmethod
    def parse_verdict(grading_response: str) -> str:
        verdict_line = _extract_verdict_line(grading_response)
        match = re.search(r"VERDICT:\s*(CORRECT|INCORRECT)\s*$", verdict_line)
        if not match:
            return "INCORRECT"
        return match.group(1)

    def grade_sample(self, question: str, target: str, predicted_answer: str) -> str:
        if self.grader_model is None:
            raise ValueError("grader_model is required for FrontierScience Olympiad")

        grader_prompt = (
            OLYMPIAD_JUDGE_PROMPT.replace("{problem}", question)
            .replace("{reference answer}", target)
            .replace("{answer}", predicted_answer)
        )
        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        sampler_response = self.grader_model(prompt_messages)
        return sampler_response.response_text

    def evaluate(self, generation_data: List[Dict]) -> List[SingleEvalResult]:
        if self.grader_model is None:
            raise ValueError("grader_model is required for FrontierScience Olympiad")
        if not generation_data:
            return []

        def evaluate_single(gen_data: Dict) -> SingleEvalResult:
            row = gen_data["row"]
            response_text = gen_data["response_text"]
            grader_response = self.grade_sample(row["problem"], row["answer"], response_text)
            verdict = self.parse_verdict(grader_response)
            score = 1.0 if verdict == "CORRECT" else 0.0
            return SingleEvalResult(
                id=row["id"],
                score=score,
                metrics={"accuracy": score},
                gt_answer=row["answer"],
                pred_answer=response_text,
                example_level_metadata={
                    "grader_response": grader_response,
                    "verdict": verdict,
                },
            )

        return common.map_with_progress(evaluate_single, generation_data, num_threads=5)


class FrontierScienceResearchEval(Eval):
    def __init__(self, grader_model: SamplerBase | None = None):
        self.grader_model = grader_model

    @staticmethod
    def parse_verdict(grading_response: str) -> float:
        verdict_line = _extract_verdict_line(grading_response)
        match = re.search(r"VERDICT:\s*(-?\d+(?:\.\d+)?)\s*$", verdict_line)
        if not match:
            return 0.0
        return min(10.0, max(0.0, float(match.group(1))))

    def grade_sample(self, question: str, rubric: str, predicted_answer: str) -> str:
        if self.grader_model is None:
            raise ValueError("grader_model is required for FrontierScience Research")

        grader_prompt = (
            RESEARCH_JUDGE_PROMPT.replace("{problem}", question)
            .replace("{rubric}", rubric)
            .replace("{answer}", predicted_answer)
        )
        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        sampler_response = self.grader_model(prompt_messages)
        return sampler_response.response_text

    def evaluate(self, generation_data: List[Dict]) -> List[SingleEvalResult]:
        if self.grader_model is None:
            raise ValueError("grader_model is required for FrontierScience Research")
        if not generation_data:
            return []

        def evaluate_single(gen_data: Dict) -> SingleEvalResult:
            row = gen_data["row"]
            response_text = gen_data["response_text"]
            grader_response = self.grade_sample(row["problem"], row["answer"], response_text)
            raw_points = self.parse_verdict(grader_response)
            accuracy = 1.0 if raw_points >= 7.0 else 0.0 # following official eval
            return SingleEvalResult(
                id=row["id"],
                score=accuracy,
                metrics={
                    "accuracy": accuracy,
                    "points": raw_points,
                },
                gt_answer=row["answer"],
                pred_answer=response_text,
                example_level_metadata={
                    "grader_response": grader_response,
                    "verdict": raw_points,
                },
            )

        return common.map_with_progress(evaluate_single, generation_data, num_threads=2)
