import argparse
import yaml
import json
import os
import random
import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional, TypedDict
from samplers import common
from samplers._types import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult
from dr_agent.dataset_utils.load_dataset import load_dataset
from dotenv import load_dotenv

load_dotenv()


# Questions were derived from Supplemental Data S2 of https://pmc.ncbi.nlm.nih.gov/articles/PMC12120168/#app2

EVAL_DIR = Path(__file__).parent.parent

class RubricItem(TypedDict):
    name: str
    criterion: str
    points: Optional[float] = 1.0

class QuestionType(TypedDict):
    name: str
    template: str
    rubrics: list[RubricItem]

class Example(TypedDict):
    variant: str
    question_type: str
    rubrics: list[RubricItem]


SYSTEM_MESSAGE = "You are a helpful research assistant that specializes in clinical genetics."

GRADER_TEMPLATE = """
Your job is to look at a research report and a rubric item, and score the response according to the criteria specified in the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

For example, if the conversation is "user: what is the pathomechanism of variant xyz? assistant: the pathomechanism is loss-of-function" and the rubric item is "mentions the pathomechanism is gain-of-function", you should return a json like this:

```json
{
  "explanation": "The assistant did not mention the pathomechanism is gain-of-function.",
  "criteria_met": false
}
```

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


# for a given citation, get the number of claims in the response (that cited it) that are supported by it.
CITATION_GRADER_TEMPLATE = """
# Instructions
You are a claim validator. You will be given the text content from a webpage, and 
a list of claims from a research report that cited the webpage.

<CONTENT OF CITED SOURCE>
<<content>>
</CONTENT OF CITED SOURCE>

<PROMPT USED TO GENERATE REPORT>
<<prompt>>
</PROMPT USED TO GENERATE REPORT>

<CLAIMS FROM GENERATED REPORT>
<<claims>>
</CLAIMS FROM GENERATED REPORT>

For each claim given you, you will determine if it is supported by the original source's content given. For source content
with only the title available, judge them as `supporting` if the title indicates that the
paper is likely relevant to the claim being considered.
Return a JSON object with a single key `claims` which is a list of `claim` objects,
one for each phrase given from the research report. Each `claim` object contains the
claim itself (`text`), a boolean `is_supported` which indicates if the claim is fully
supported by the citation source content, and `explanation`, which should be a string briefly
explaining why the response is or is not supported by the citation source content.
If the provided source content is completely empty, entirely filler webpage text, or is an error message or content filter, 
output an empty list for the `claims` key.

# Example

USER:
Content of cited source: "US holidays include Veteran's Day (November 11th) and Halloween (October 31st)."

Claims from generated report:
- "Veteran's Day is a national holiday in November."

ASSISTANT:
```json
"claims": [
    {
      "text": "Veteran's Day is a national holiday in November.",
      "is_supported": true,
      "explanation": "Sources states that Veteran's Day falls on November 11th."
    },
]
```

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
"""


def get_citation_map(gen_data: dict, run_mode: str) -> dict[str, str]:
    print(f"Using run mode {run_mode} to process citations")
    if run_mode == "auto_search_sft" or run_mode == "auto_reason_search":
        full_trace = gen_data["full_traces"]["generated_text"]
        # generated texts contain direct tool outputs as <snippet>...</snippet> tags
        search_pattern = re.compile(r'<snippet\s+id=([a-zA-Z0-9\-]+)>(.*?)</snippet>', re.DOTALL)
        snippet_map = {match[0]: match[1].strip() for match in search_pattern.findall(full_trace)}
        webpage_pattern = re.compile(r'<webpage\s+id=([a-zA-Z0-9\-]+)>(.*?)</webpage>', re.DOTALL)
        webpage_map = {match[0]: match[1].strip() for match in webpage_pattern.findall(full_trace)}
        snippet_map.update(webpage_map)

        response_text = gen_data["response_text"]
        citation_map = {}
        pattern = r'<cite\s+id="([^"]+)">(.*?)</cite>'
        for match in re.finditer(pattern, response_text):
            cite_id_str = match.group(1)
            span = match.group(2)
            ids = [cid.strip() for cid in cite_id_str.split(",")]

            # assign the span to each individual id
            for cite_id in ids:
                if not cite_id in citation_map:
                    citation_map[cite_id] = {
                        "original_source": snippet_map.get(cite_id, "This citation id was hallucinated. All claims that cite it are unsupported."),
                        "cited_spans": ""
                    }
                citation_map[cite_id]["cited_spans"] += span + "\n\n"
        return citation_map
    
    elif run_mode in ["oai_search", "gemini_search", "asta"]:
        return gen_data["full_traces"]["cited_snippets"]

    raise ValueError(f"Invalid run mode: {run_mode}")


class GeneticDiseasesEval(Eval):
    def __init__(
        self,
        grader_model: SamplerBase,
        run_mode: str,
        n_threads: int = 10,
        generation_only: bool = True,
    ):
        self.grader_model = grader_model
        self.run_mode = run_mode
        self.generation_only = generation_only
        self.n_threads = n_threads

    def generate(self, sampler: SamplerBase) -> list[dict]:
        """Generate responses for all examples and return raw generation data"""

        def generate_single(row: dict):
            # prompt is derived from the question type template and instance-specific variant
            question_type = row["question_type"]
            variant = row["variant"]
            prompt = self.question_types[question_type]["template"].format(variant=variant)
            prompt_messages: MessageList = [dict(content=prompt, role='user')]
            row["prompt"] = prompt

            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            response_dict = sampler_response.response_metadata
            actual_queried_prompt_messages = (
                sampler_response.actual_queried_message_list
            )
            response_usage = response_dict.get("usage", None)
            full_traces = sampler_response.full_traces

            output = {
                "row": row,
                "response_text": response_text,
                "actual_queried_prompt_messages": actual_queried_prompt_messages,
                "response_usage": response_usage,
                "full_traces": full_traces,
            }
            return output

        generation_data = common.map_with_progress(
            generate_single,
            self.examples,
            num_threads=self.n_threads,
            pbar=True,
        )

        return generation_data

    def evaluate(self, generation_data: list[dict], results_path: str = None) -> list[SingleEvalResult]:
        """Evaluate generated responses and return SingleEvalResult objects"""
        if self.grader_model is None:
            raise ValueError(
                "grader_model is required for evaluation. Set eval_obj.grader_model before calling evaluate()."
            )

        # Temporarily disable generation_only mode for evaluation
        self.generation_only = False

        if results_path is not None and os.path.exists(results_path):
            print(f"Results already exist at {results_path}")
            with open(results_path, "r") as f:
                results_data = json.load(f)
            single_eval_results = results_data["per_example_results"]
            return single_eval_results

        dataset = load_dataset({"name": "genetic_diseases_qa"})
        dataset_dict = {ele["id"]: ele for ele in dataset}

        def evaluate_single(gen_data: dict) -> SingleEvalResult:
            import sys
            row = gen_data["row"]
            response_text = gen_data["response_text"]
            actual_queried_prompt_messages = gen_data["actual_queried_prompt_messages"]

            # in case that full trace errored, the response text will contain the error message
            if gen_data.get("full_traces") is None:
                gen_data["full_traces"] = {"generated_text": response_text}

            # get cited snippets from the full_trace that appear in the response
            citation_map = get_citation_map(gen_data, self.run_mode)
            print(f"Found {len(citation_map)} cited snippets")

            # add a short preview of each cited snippet to the response, but
            # the citation scores will use the full version
            if len(citation_map):
                response_text += "\n\nCited texts:\n"
                for sid,snippet in citation_map.items():
                    preview = snippet
                    if isinstance(snippet, dict):
                        preview = snippet.get("original_source", "")
                    preview = preview[:100] + "..."
                    response_text += f"[{sid}]: {preview}\n\n"

            # gets both general and instance-specific rubrics
            rubric_items = dataset_dict[row["id"]]["rubrics"]
            renamed_rubrics = []
            for r in rubric_items:  # hacky way to handle old rubric names
                if r["name"] == "correct_answer_detail":
                    r["name"] = "correct_answer"
                renamed_rubrics.append(r)
            rubric_items = [RubricItem(**d) for d in renamed_rubrics]

            metrics, readable_explanation_str, rubric_items_with_grades = (
                self.grade_sample(
                    prompt=actual_queried_prompt_messages,
                    response_text=response_text,
                    rubric_items=rubric_items,
                    example_tags=[],
                    citation_map=citation_map,
                )
            )
            print(readable_explanation_str)

            score = metrics["overall_score"]

            return SingleEvalResult(
                id=row["id"],
                score=score,
                metrics=metrics,
                pred_answer=response_text,
            )
        
        results = common.map_with_progress(
            evaluate_single,
            generation_data,
            num_threads=self.n_threads,
            pbar=True,
        )

        return results
    

    def grade_sample(
        self,
        prompt: list[dict[str, str]],
        response_text: str,
        example_tags: list[str],
        rubric_items: list[RubricItem],
        citation_map: dict[str, str],
    ) -> tuple[dict, str, list[dict]]:
        for i in range(len(rubric_items)):
            rubric_items[i]["points"] = rubric_items[i].get("points", 1.0)
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
            # NOTE: `name` is used instead of `tags` for the rare variants data
            rubric_tag_scores = {}
            for rubric_item in rubric_items:
                rubric_tag_scores[rubric_item["name"]] = 0.0
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
            # only include cited texts for consistency criterion
            convo_str = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in convo_with_response]
            )
            role2message = {m["role"]: m["content"] for m in convo_with_response}
            user_query = role2message["user"]

            if rubric_item["name"] == "consistency":
                claims_supported = 0
                total_claims = 0
                errors = 0
                graded_claims_list = []
                for snippet_id, payload in citation_map.items():
                    content = payload["original_source"]
                    if len(content) > 20000:
                        # avoid sending ridiculously long webpage content to the grader
                        content = content[:20000] + "..."
                    cite_spans = payload["cited_spans"].replace("Parts of the response that reference this citation:\n\n", "")
                    grader_prompt = CITATION_GRADER_TEMPLATE.replace(
                        "<<content>>", content
                    ).replace("<<prompt>>", user_query).replace("<<claims>>", cite_spans)
                    messages: MessageList = [dict(content=grader_prompt, role="user")]
                    retries = 1
                    # TODO: better error handling, this is fragile
                    while retries > 0:
                        sampler_response = self.grader_model(messages)
                        grading_response = sampler_response.response_text
                        grading_response_dict = parse_json_to_dict(grading_response)
                        if "claims" in grading_response_dict and isinstance(grading_response_dict["claims"], list):
                            if all("is_supported" in c for c in grading_response_dict["claims"]):
                                num_supported = sum(1 for c in grading_response_dict["claims"] if c["is_supported"])
                                claims_supported += num_supported
                                total_claims += len(grading_response_dict["claims"])
                                graded_claims_list.extend(grading_response_dict["claims"])
                                break
                        retries -= 1
                        print("Grading failed due to bad JSON output, retrying...")
                    if retries == 0:
                        errors += 1
                explanation = "\n".join([f"- {c['text']}: {c['explanation']}" for c in graded_claims_list])
                return {
                    "claims_supported": claims_supported,
                    "total_claims": total_claims,
                    "total_citations": len(citation_map),
                    "grading_errors": errors,
                    "explanation": explanation,
                }

            # otherwise, grade as binary rubric            
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
            tag = rubric_item["name"]
            rubric_tag_items_grades[tag].append((rubric_item, grading_response))
        

        rubric_tag_scores = {}
        for tag, items_grades in rubric_tag_items_grades.items():
            assert tag in ["correct_answer", "correct_answer_detail", "consistency", "synthesis", "evidence_quality"], f"Unexpected tag {tag}"
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
            if rubric_item["name"] == "consistency":
                criteria_met = grading_response["total_claims"] > 0 and grading_response["claims_supported"] / grading_response["total_claims"] > 0.5
                readable_explanation = (
                    f"[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}"
                )
                readable_explanation_list.append(readable_explanation)
                rubric_items_with_grades.append({
                    **dict(rubric_item),
                    **grading_response
                })
            else:
                criteria_met = grading_response["criteria_met"]
                readable_explanation = (
                    f"[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}"
                )
                readable_explanation_list.append(readable_explanation)
                rubric_items_with_grades.append(
                    {
                        **dict(rubric_item),
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

    def __call__(self, sampler: SamplerBase) -> EvalResult:
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
                        id=gen_data["row"]["example_id"],
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

        agg_results = common.aggregate_results(results)
        return agg_results




def calculate_score(
    rubric_items: list[RubricItem], grading_response_list: list[dict]
) -> float | None:
    total_possible_points = len(rubric_items)
    if total_possible_points == 0:
        # should not happen for overall score, but may happen for tags
        return None
    
    achieved_points = 0
    for grading_response in grading_response_list:
        if "claims_supported" in grading_response and grading_response["total_claims"] > 0:
            cite_acc = grading_response["claims_supported"] / grading_response["total_claims"]
            achieved_points += cite_acc
        elif "criteria_met" in grading_response and grading_response["criteria_met"]:
            achieved_points += 1
    
    overall_score = achieved_points / total_possible_points
    return overall_score

def parse_json_to_dict(json_string: str) -> dict:
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {}

