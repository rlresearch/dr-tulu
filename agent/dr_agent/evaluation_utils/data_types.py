from typing import Any, Dict, List, Optional, TypedDict, Union


class DatasetConfig(TypedDict):
    """Configuration for loading evaluation datasets."""

    name: str  # Required: Dataset name ('browsecomp', 'simpleqa', 'healthbench', etc.)
    num_examples: Optional[int]  # Optional: Number of examples to load (None for all)

    subset: Optional[str]
    # Optional: Dataset subset (e.g., 'hard', 'consensus' for healthbench)


class DatasetInstance(TypedDict):
    id: str
    problem: str


class EvalOutput(TypedDict):
    example_id: str
    problem: str
    final_response: str
    full_traces: Dict[str, Any]
    original_data: Optional[Dict[str, Any]]
