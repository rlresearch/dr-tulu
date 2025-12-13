import json
from dataclasses import dataclass, field
from typing import Any, Literal, overload

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]



@dataclass
class SamplerResponse:
    """
    Response from a sampler.
    """
    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]
    full_traces: Any = None

class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    def __call__(
        self, 
        message_list: MessageList,
    ) -> SamplerResponse:
        raise NotImplementedError


class HealthBenchSampler(SamplerBase):
    """
    Sample from HealthBench
    """

    def __init__(self, results_path: str):
        self.results_path = results_path
        self.generation_results = []
        with open(results_path, "r") as f:
            for line in f:
                self.generation_results.append(json.loads(line))

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        for result in self.generation_results:
            if result.get("prompt") == message_list:
                response = SamplerResponse(
                    response_text=result.get("response_text", ""),
                    actual_queried_message_list=message_list,
                    response_metadata={},
                )
                # Only include full_traces if it exists in the result
                if "full_traces" in result:
                    response.full_traces = result["full_traces"]
                return response
        raise ValueError("No matching response found for the given message list.")

@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: float | None  # top-line metric
    metrics: dict[str, float] | None  # other metrics
    htmls: list[str] | None = None # strings of valid HTML
    convos: list[MessageList] | None = None # sampled conversations
    metadata: dict[str, Any] | None = None  # Extra data such as rubric scores or sollen
    per_example_results: dict[str, Any] | None = None  # Per-example results

@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """
    id: str
    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None  # sampled conversation
    example_level_metadata: dict[str, Any] | None = (
        None  # Extra data such as rubric scores or sollen
    )
    gt_answer: str | None = None
    pred_answer: str | None = None


class Eval:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError
