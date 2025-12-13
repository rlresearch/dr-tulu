import json
import os
import shutil
from dataclasses import dataclass
from typing import List

from huggingface_hub import hf_hub_download


@dataclass
class RubricItem:
    rubric_item: str
    type: str


@dataclass
class ResearchQAItem:
    id: str
    general_domain: str
    subdomain: str
    field: str
    query: str
    date: str
    rubric: List[RubricItem]


def download_researchqa_dataset(
    split: str = "test.json", output_dir: str = "evaluation/research_qa_eval/data"
) -> str:
    output_path = os.path.join(output_dir, split)

    if not os.path.exists(output_path):
        os.makedirs(output_dir, exist_ok=True)
        file_path = hf_hub_download(
            repo_id="realliyifei/ResearchQA",
            filename=split,
            repo_type="dataset",
            revision="87cdd81df0c5ea96de293859233e8e64dac3d168",
        )
        shutil.copy(file_path, output_path)

    return output_path


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
