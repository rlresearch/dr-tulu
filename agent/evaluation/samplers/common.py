import io
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
from typing import Any, Callable

import jinja2
import numpy as np
import requests
from tqdm import tqdm

from ._types import EvalResult, Message, SamplerBase, SingleEvalResult

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    "Answer\s*:",
    "Answer\s*:​​​​​​",  # Korean invisible character
    "উত্তর\s*:",
    "उत्तर\s*:",
    "উত্তরঃ",
    "উত্তর\s*:",
    "Antwort\s*:",
    "답변\s*:",
    "정답\s*:",
    "답\s*:",
    "答案\s*：",
    "答案\s*:",
    "答\s*：",
    "答\s*:",
    "答复\s*：",
    "答曰\s*：",
    "الإجابة:",
    "الجواب:",
    "إجابة:",
    "الإجابة النهائية:",
    "الإجابة الصحيحة:",
    "الإجابة الصحيحة هي:",
    "الإجابة هي:",
    "الجواب النهائي:",
    "Respuesta\s*:",
    "Risposta\s*:",
    "答え\s*:",
    "答え\s*：",
    "回答\s*:",
    "回答\s*：",
    "解答\s*:",
    "Jawaban\s*:",
    "Réponse\s*:",
    "Resposta\s*:",
    "Jibu\s*:",
    "Idahun\s*:",
    "Ìdáhùn\s*:",
    "Idáhùn\s*:",
    "Àmọ̀nà\s*:",
    "Àdáhùn\s*:",
    "Ànúgọ\s*:",
    "Àṣàyàn\s*:",
]


EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }}</p>
<p>Extracted Answer: {{ extracted_answer }}</p>
<p>Score: {{ score }}</p>
"""


def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def check_equality(sampler: SamplerBase, expr1: str, expr2: str):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    sampler_response = sampler([dict(content=prompt, role="user")])
    response_text = sampler_response.response_text
    return response_text.lower().strip() == "yes"


def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        return np.std(
            [np.mean(np.random.choice(values, len(values))) for _ in range(1000)]
        )
    else:
        raise ValueError(f"Unknown {stat =}")


def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str, ...] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] | None = None,
) -> EvalResult:
    """
    Aggregate results from multiple evaluations into a single EvalResult.
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    full_traces_list = []
    per_example_results = [] 
    
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        per_example_results.append(single_eval_result.__dict__)
        # htmls.append(single_eval_result.html)
        # convos.append(single_eval_result.convo)
        # metadata.append(single_eval_result.example_level_metadata)
        
        # # Collect full_traces if available
        # if (single_eval_result.example_level_metadata and 
        #     "full_traces" in single_eval_result.example_level_metadata):
        #     full_traces_list.append(single_eval_result.example_level_metadata["full_traces"])
        # else:
        #     full_traces_list.append(None)
    
    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)
    
    # Include full_traces in metadata
    # result_metadata = {
    #     "example_level_metadata": metadata,
    #     "full_traces": full_traces_list
    # }
    
    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        per_example_results=per_example_results,
        # htmls=htmls,
        # convos=convos,
        # metadata=result_metadata,
    )


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap(f, xs), total=len(xs)))


def map_with_progress_checkpoint(
    f: Callable,
    xs: list[Any],
    checkpoint_path: str,
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
    checkpoint_interval: int = 10,
):
    """
    Apply f to each element of xs with incremental saving of results.
    Saves conversations, traces, and responses separately to avoid serialization issues.
    """
    import json
    import os

    # Define separate checkpoint files
    base_path = checkpoint_path.replace('.checkpoint.json', '')
    convos_checkpoint = f"{base_path}_convos.checkpoint.jsonl"
    traces_checkpoint = f"{base_path}_traces.checkpoint.json"
    responses_checkpoint = f"{base_path}_responses.checkpoint.json"
    
    # Load existing results if checkpoints exist
    completed_results = []
    start_idx = 0
    
    # Check if we have existing checkpoint data
    if os.path.exists(responses_checkpoint):
        try:
            with open(responses_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)
                completed_results = checkpoint_data.get("results", [])
                start_idx = len(completed_results)
                if start_idx > 0:
                    print(f"[CHECKPOINT] Resuming from example {start_idx + 1}/{len(xs)}")
        except (json.JSONDecodeError, KeyError):
            print(f"[CHECKPOINT] Corrupted checkpoint, starting fresh")
            start_idx = 0
            completed_results = []
    
    # If already completed, clean up checkpoints and return results
    if start_idx >= len(xs):
        print(f"[CHECKPOINT] All examples already completed ({len(completed_results)} results)")
        # Clean up checkpoint files
        for checkpoint_file in [convos_checkpoint, traces_checkpoint, responses_checkpoint]:
            if os.path.exists(checkpoint_file):
                os.unlink(checkpoint_file)
                print(f"[CHECKPOINT] Cleaned up {checkpoint_file}")
        return completed_results
    
    # Process remaining examples
    remaining_xs = xs[start_idx:]
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x
    
    def save_incremental_results(results_so_far):
        """Save results incrementally to separate files"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Save basic responses data
        responses_data = {
            "total_examples": len(xs),
            "completed_examples": len(results_so_far),
            "results": results_so_far
        }
        with open(responses_checkpoint, 'w') as f:
            json.dump(responses_data, f, indent=2, default=str)
        
        # Save conversations separately
        conversations = []
        traces = []
        
        for result in results_so_far:
            # Extract conversation
            if "actual_queried_prompt_messages" in result:
                convo = result["actual_queried_prompt_messages"] + [
                    {"role": "assistant", "content": result.get("response_text", "")}
                ]
                conversations.append(convo)
            elif "actual_queried_message_list" in result:
                convo = result["actual_queried_message_list"] + [
                    {"role": "assistant", "content": result.get("response_text", "")}
                ]
                conversations.append(convo)
            
            # Extract traces
            if result.get("full_traces"):
                traces.append(result["full_traces"])
        
        # Save conversations
        if conversations:
            with open(convos_checkpoint, 'w') as f:
                for convo in conversations:
                    f.write(json.dumps(convo, default=str) + "\n")
        
        # Save traces
        if traces:
            with open(traces_checkpoint, 'w') as f:
                json.dump(traces, f, indent=2, default=str)
    
    if os.getenv("debug"):
        # Sequential processing with checkpoints
        for i, x in enumerate(pbar_fn(remaining_xs, total=len(remaining_xs))):
            result = f(x)
            completed_results.append(result)
            
            # Save checkpoint every N examples
            if (len(completed_results)) % checkpoint_interval == 0:
                save_incremental_results(completed_results)
                print(f"[CHECKPOINT] Saved progress: {len(completed_results)}/{len(xs)} examples")
    else:
        # Parallel processing with periodic checkpoints
        with ThreadPool(min(num_threads, len(remaining_xs))) as pool:
            batch_size = checkpoint_interval
            
            for i in range(0, len(remaining_xs), batch_size):
                batch = remaining_xs[i:i+batch_size]
                batch_results = list(pool.map(f, batch))
                completed_results.extend(batch_results)
                
                # Save checkpoint after each batch
                save_incremental_results(completed_results)
                current_total = start_idx + i + len(batch)
                print(f"[CHECKPOINT] Saved progress: {current_total}/{len(xs)} examples")
    
    # Final save and cleanup
    save_incremental_results(completed_results)
    
    # If we've completed all examples, clean up the checkpoint files
    if len(completed_results) >= len(xs):
        for checkpoint_file in [convos_checkpoint, traces_checkpoint, responses_checkpoint]:
            if os.path.exists(checkpoint_file):
                os.unlink(checkpoint_file)
                print(f"[CHECKPOINT] Completed! Cleaned up {checkpoint_file}")
    
    return completed_results


jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)
_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""


def message_to_html(message: Message) -> str:
    """
    Generate HTML snippet (inside a <div>) for a message.
    """
    return jinja_env.from_string(_message_template).render(
        role=message["role"],
        content=message["content"],
        variant=message.get("variant", None),
    )


jinja_env.globals["message_to_html"] = message_to_html


_report_template = """<!DOCTYPE html>
<html>
    <head>
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html | safe }}
    <hr>
    {% endfor %}
    </body>
</html>
"""


def make_report(eval_result: EvalResult) -> str:
    """
    Create a standalone HTML report from an EvalResult.
    """
    return jinja_env.from_string(_report_template).render(
        score=eval_result.score,
        metrics=eval_result.metrics,
        htmls=eval_result.htmls,
    )


def make_report_from_example_htmls(htmls: list[str]):
    """
    Create a standalone HTML report from a list of example htmls
    """
    return jinja_env.from_string(_report_template).render(
        score=None, metrics={}, htmls=htmls
    )


def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


def url_to_fileobj(url: str, binary=False) -> Any:
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content) if binary else io.StringIO(response.text)


def has_only_user_assistant_messages(messages: list[Message]) -> bool:
    """
    Check if the messages only contain user and assistant messages.
    """
    return all(m["role"] in ("user", "assistant") for m in messages)