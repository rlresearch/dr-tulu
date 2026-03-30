"""
Chat template utilities for Rubrics RL training.

This module provides a registry-based system for formatting prompts:
1. Policy training (normal system prompt + user question)
2. Rubric generation (system prompt for rubric generation + user question)
3. Reward/judge evaluation (system prompt + rubric + question + answer)

Each registry entry returns a list of messages and optionally tokenizes them.
"""

from collections.abc import Callable
from typing import Any

from transformers import PreTrainedTokenizer

# Default system prompts
DEFAULT_POLICY_SYSTEM_PROMPT = "You are a helpful assistant."

# Original rubric generation prompt (pre-2026-03-04). Kept for reproducibility.
V0_RUBRIC_GENERATION_SYSTEM_PROMPT = """You are an expert evaluator generating rubrics to assess answers to questions.

Given a question, generate a comprehensive rubric that can be used to evaluate answers. The rubric should:
1. Identify key criteria for a good answer
2. Be specific and measurable
3. Cover the most important aspects of answering the question
4. Include explicit numeric scores for each criterion (e.g., "Score: 0.3" or "Worth 0.25 points")
5. Ensure all scores sum to exactly 1.0

Always include numeric values in your rubric. Generate the rubric directly without JSON formatting."""

DEFAULT_RUBRIC_GENERATION_SYSTEM_PROMPT = """You are an expert evaluator generating rubrics to assess answers to questions.

Given a question, first analyze it to identify:
- Explicit requirements: directly stated constraints, formatting rules, or content directives (e.g., "list three reasons", "write in Python", "under 100 words")
- Implicit requirements: unstated but necessary qualities inferred from context (e.g., explaining "blockchain to grandparents" implicitly requires avoiding jargon, even if not explicitly forbidden)

Then generate a rubric of 5-10 criteria following these rules:
1. Structural atomicity: each criterion targets exactly one aspect. Do not combine multiple conditions into one criterion.
2. Semantic objectivity: write criteria based only on the question, without assuming any specific answer.
3. All weights must sum to exactly 1.0, reflecting each criterion's importance.

For each criterion should be described in a sentence, define scoring levels from 0.0 to 1.0. At minimum include 1.0 and 0.0, and add intermediate levels (e.g., 0.5, 0.3, 0.8) wherever useful for distinguishing answer quality. A judge will score each criterion and multiply by the weight to produce a total score.

Output ONLY valid JSON in this format:
{"criteria": [{"criterion": "<a sentence of what this criterion measures>", "weight": <float>, "scoring_levels": {"1.0": "<description>", "0.5": "<description>", "0.0": "<description>"}}, {"criterion": ...}, ...]}

Example — question: "Explain how photosynthesis works in simple terms"
{"criteria": [{"criterion": "Explains that plants convert sunlight into chemical energy", "weight": 0.3, "scoring_levels": {"1.0": "Clearly explains the sunlight-to-energy conversion", "0.5": "Mentions sunlight or energy but not the conversion process", "0.0": "No mention of the energy conversion mechanism"}}, {"criterion": "Identifies CO2 and water as inputs and oxygen and glucose as outputs", "weight": 0.25, "scoring_levels": {"1.0": "All four substances correctly identified", "0.5": "Some inputs or outputs mentioned but incomplete", "0.0": "None identified or incorrect"}}, {"criterion": "Avoids unnecessary jargon and is understandable to a general audience", "weight": 0.2, "scoring_levels": {"1.0": "Clear and jargon-free throughout", "0.5": "Mostly accessible but uses some unexplained technical terms", "0.0": "Dense with jargon, inaccessible to a general reader"}}, {"criterion": "Follows a coherent structure from inputs to process to outputs", "weight": 0.15, "scoring_levels": {"1.0": "Well-organized with clear progression", "0.5": "Some structure but jumps between ideas", "0.0": "Disorganized or incoherent"}}, {"criterion": "All stated facts about photosynthesis are correct", "weight": 0.1, "scoring_levels": {"1.0": "No factual errors", "0.5": "Minor inaccuracy that does not undermine the explanation", "0.0": "Contains significant factual errors"}}]}"""


V3_RUBRIC_GENERATION_SYSTEM_PROMPT = """You are an expert evaluator generating rubrics to assess answers to questions.

Given a question, first analyze it to identify:
- First identify the most important aspect of the question that the answer should satisfy, this will be used to form the Dealbreaker criterion (explained later).
- Explicit requirements: directly stated constraints, formatting rules, or content directives (e.g., "list three reasons", "write in Python", "under 100 words")
- Implicit requirements: unstated but necessary qualities inferred from context (e.g., explaining "blockchain to grandparents" implicitly requires avoiding jargon, even if not explicitly forbidden)

Then generate a rubric of 2-5 criteria following these rules:
1. Structural atomicity: each criterion targets exactly one aspect. Do not combine multiple conditions into one criterion.
2. Semantic objectivity: write criteria based only on the question, without assuming any specific answer.
3. All weights must sum to exactly 1.0, reflecting each criterion's importance, important criteria such as accuracy should have higher weight.
4. Dealbreaker criterion: some criteria are so important that if the answer does not satisfy them, the answer is just not good enough. For example, for questions with verifiable short form answers or multiple choice (such as math or factuality), the dealbreaker criterion should be "the answer is equivalent to XXX (e.g. 100)". A Dealbreaker criterion should have very high weight such as 0.8. But form explanation based or questions requiring long form answers, accuracy should not have such a high weight because it's too general and it will be hard for the judge to assess accuracy. It's fine if there's no dealbreaker.

For each criterion should be described in a sentence, define scoring levels from 0.0 to 1.0. At minimum include 1.0 and 0.0, and add intermediate levels (e.g., 0.5, 0.3, 0.8) wherever useful for distinguishing answer quality. A judge will score each criterion and multiply by the weight to produce a total score.

Output ONLY valid JSON in this format:
{"criteria": [{"criterion": "<a sentence of what this criterion measures>", "weight": <float>, "scoring_levels": {"1.0": "<description>", "0.5": "<description>", "0.0": "<description>"}}, {"criterion": ...}, ...]}

Example — question: "Explain how photosynthesis works in simple terms"
{"criteria": [{"criterion": "Explains that plants convert sunlight into chemical energy", "weight": 0.3, "scoring_levels": {"1.0": "Clearly explains the sunlight-to-energy conversion", "0.7": "Mentions sunlight or energy but not the conversion process", "0.3": "Vague reference to energy without clear connection to sunlight", "0.0": "No mention of the energy conversion mechanism"}}, {"criterion": "Identifies CO2 and water as inputs and oxygen and glucose as outputs", "weight": 0.25, "scoring_levels": {"1.0": "All four substances correctly identified", "0.6": "Three substances correctly identified", "0.3": "Some inputs or outputs mentioned but incomplete", "0.0": "None identified or incorrect"}}, {"criterion": "Avoids unnecessary jargon and is understandable to a general audience", "weight": 0.2, "scoring_levels": {"1.0": "Clear and jargon-free throughout", "0.8": "Mostly accessible with minimal technical terms that are explained", "0.4": "Mostly accessible but uses some unexplained technical terms", "0.0": "Dense with jargon, inaccessible to a general reader"}}, {"criterion": "Follows a coherent structure from inputs to process to outputs", "weight": 0.15, "scoring_levels": {"1.0": "Well-organized with clear progression", "0.6": "Generally structured but minor organizational issues", "0.2": "Some structure but jumps between ideas", "0.0": "Disorganized or incoherent"}}, {"criterion": "All stated facts about photosynthesis are correct", "weight": 0.1, "scoring_levels": {"1.0": "No factual errors", "0.8": "Minor inaccuracy that does not undermine the explanation", "0.0": "Contains significant factual errors"}}]}"""


CORRECTNESS_FOCUSED_RUBRIC_GENERATION_SYSTEM_PROMPT = """You are an expert evaluator. Given a question, describe what makes a correct and complete answer.

Focus on:
1. The key facts, reasoning steps, or solution elements that a correct answer MUST include
2. Common mistakes or misconceptions to watch for
3. How to distinguish a correct answer from a plausible but wrong one

Be concise (under 500 words). Do NOT use multi-criteria weighted rubrics. Instead, describe the essential correctness requirements as a focused checklist."""

DEFAULT_RUBRIC_GENERATION_WITH_EXAMPLES_SYSTEM_PROMPT = "You are an expert educator. Given a question, generate a scoring rubric that can effectively distinguish between high-quality and lower-quality answers."

DEFAULT_JUDGE_SYSTEM_PROMPT = "You are an expert evaluator judging answers based on a rubric."
DEFAULT_BINARY_JUDGE_SYSTEM_PROMPT = (
    "You are a judge, evaluating whether a response satisfies the given rubric. "
    "If the response satisfies the criterion of the rubric, output YES; otherwise output NO."
)
DEFAULT_RLCER_VERIFIER_SYSTEM_PROMPT = (
    "You are an expert in evaluating student responses to math problems. "
    "Your task is to assess a given response based on a set of binary rubrics and compute a final score."
)


def _binary_judge_messages(content: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_BINARY_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Requirement:
- You must follow the rubric strictly, and only consider the criteria listed in the rubric.
- You must NOT consider any other factors, such as your own opinions or external knowledge.

Below between <RESPONSE> and </RESPONSE> is the response to evaluate:
<RESPONSE>{content["answer"]}</RESPONSE>

Below between <RUBRIC> and </RUBRIC> is the rubric to evaluate on:
<RUBRIC>{content["rubric"]}</RUBRIC>

Output STRICTLY in below format. No other text is allowed:
<EVALUATION> YES/NO </EVALUATION>""",
        },
    ]


def _rlcer_verifier_messages(content: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_RLCER_VERIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""[Task Input]:
- Question: The math problem the student is solving.
- Response: The student's step-by-step solution.
- Rubrics: A list of criteria for evaluation. Each criterion has:
  - category: The aspect of reasoning being assessed.
  - criterion: A binary statement (either a merit to reward or a flaw to penalize).
  - points: The points assigned (positive for rewards, negative for penalties).

[Task Instruction]:
Evaluate each criterion:
- For each criterion in the rubrics list (in the given order), determine if the criterion is satisfied by the response.
  - Output True if the criterion is met (for a reward) or if the flaw is present (for a penalty).
  - Output False otherwise.
- Base your judgement solely on the content of the response and the specific wording of the criterion.
- Compute the final score: Sum the points of every criterion that is evaluated as True (this includes both positive and negative points).

[Output Format]
Return a JSON object with the following structure:
```json
{{
  "judgement": [boolean, boolean, ...],
  "final_score": number
}}
```

[Important Guidelines]
- Be objective: Do not consider any external knowledge beyond the provided question, response, and rubrics.
- Binary decision: Each criterion must be evaluated as strictly True or False. There are no partial credits.
- Order matters: The judgement list must exactly match the order of the rubrics provided.
- Independent evaluation: Assess each criterion separately. The presence or absence of one flaw/merit does not influence the evaluation of others unless the criterion explicitly references another.
- Penalty criteria: For a criterion with negative points, True means the flaw is present and the negative points are added. False means the flaw is absent.
- Reward criteria: For a criterion with positive points, True means the merit is present and the positive points are added. False means the merit is absent.

Now, proceed to evaluate the following:

Question:
{content["question"]}

Response:
{content["answer"]}

Rubrics:
{content["rubrics"]}

Your final output should contain only the JSON object. Do not include any additional text or explanations.""",
        },
    ]


DEFAULT_BASELINE_SYSTEM_PROMPT = (
    "You are a baseline assistant responding without access to the latest rubric improvements."
)

DEFAULT_QUESTION_INFERENCE_SYSTEM_PROMPT = """You are an expert at understanding what question someone was trying to answer based on their response.

Given an answer, infer what question the person was likely trying to answer. Be specific and output only the inferred question, nothing else."""

DEFAULT_RRD_RUBRIC_SYSTEM_PROMPT = "You are a rubric designer for an LLM-as-judge system."
DEFAULT_RRD_FILTER_SYSTEM_PROMPT = "You are an expert rubric filtering assistant."
DEFAULT_RRD_WEIGHT_SYSTEM_PROMPT = "You are assigning rubric weights for pairwise response judging."

RRD_RUBRIC_TIPS_BLOCK = """Tips for writing good rubrics:
i. MECE:
- Mutually Exclusive, Collectively Exhaustive
ii. Completeness:
- Consider all the elements you would want to include to create a perfect response and put them into the rubric.
- Include both requested facts and supporting details (justification, reasoning, logic) when needed.
iii. No overlapping:
- The same error from a model should not be punished multiple times.
iv. Diversity:
- Rubric items should include varied types of information.
- Avoid criteria that are all simple mention checks (e.g., "mentions A", "mentions B").
v. Number of rubric items:
- There is no single golden number; it varies by task and account.
- Write enough rubrics to cover all important aspects of an ideal response.
vi. Failure rate guidance:
- A good rule of thumb is that a model should fail about 50 percent of rubric items.
vii. Atomicity / Non-stacked:
- Each criterion should evaluate exactly one distinct aspect.
- Avoid bundling multiple criteria into one line (often indicated by "and").
viii. Specificity:
- Criteria should be binary (true/false) and objective.
- Avoid vague descriptions like "the response must be accurate."
ix. Self-contained:
- Each criterion should contain all information needed for evaluation.
x. Verifiable without external search:
- Criteria should be checkable without requiring external lookup."""


def _format_rrd_responses(responses: list[str]) -> str:
    if not responses:
        return "- (none)"
    return "\n".join(f"- {r}" for r in responses)


def _format_rrd_rubrics(rubrics: list[str]) -> str:
    if not rubrics:
        return "- (none)"
    return "\n".join(f"- {r}" for r in rubrics)


# Registry of message formatters
# Each formatter is a lambda that takes content dict, returns list of messages
MESSAGE_FORMATTER_REGISTRY: dict[str, Callable[[dict[str, Any]], list[dict[str, str]]]] = {
    "policy": lambda content: [
        {"role": "system", "content": DEFAULT_POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": content["question"]},
    ],
    "rubric_generation_v0": lambda content: [
        {"role": "system", "content": V0_RUBRIC_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": content["question"]},
    ],
    "rubric_generation": lambda content: [
        {"role": "system", "content": DEFAULT_RUBRIC_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": content["question"]},
    ],
    "rubric_generation_v3": lambda content: [
        {"role": "system", "content": V3_RUBRIC_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": content["question"]},
    ],
    "rubric_generation_correctness": lambda content: [
        {"role": "system", "content": CORRECTNESS_FOCUSED_RUBRIC_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": content["question"]},
    ],
    "rubric_generation_with_examples": lambda content: [
        {"role": "system", "content": DEFAULT_RUBRIC_GENERATION_WITH_EXAMPLES_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Question:
{content["question"]}

Here are two example answers for reference:
- Policy model answer (higher quality expected): {content["policy_answer"]}
- Baseline model answer (reference quality): {content["baseline_answer"]}

Return the rubric as clear, numbered criteria with scoring guidelines that sum to 1.0 and can distinguish between answers of different quality levels.""",
        },
    ],
    # Original judge prompt (pre-2026-03-04). Kept for reproducibility of earlier experiments.
    "judge_v1": lambda content: [
        {"role": "system", "content": DEFAULT_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Question: {content["question"]}

Rubric: {content["rubric"]}

Answer to evaluate: {content["answer"]}

Evaluate the answer based on the rubric. Go through each portion of the rubric and assign a score for each portion following the grading criteria. Then add up the scores to get the total score.

Output ONLY valid JSON in the following format:
{{
  "reasoning": "<brief explanation of your evaluation>",
  "score": <total score (a float between 0.0 to 1.0)>
}}

Example outputs:
{{"reasoning": "Answer addresses all key points in the rubric with accurate details", "score": 0.9}}
{{"reasoning": "Answer partially satisfies the rubric but missing some critical elements", "score": 0.5}}
{{"reasoning": "Answer does not meet the rubric criteria", "score": 0.1}}

Your evaluation:""",
        },
    ],
    "judge": lambda content: [
        {"role": "system", "content": DEFAULT_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Question: {content["question"]}

Rubric: {content["rubric"]}

Answer to evaluate: {content["answer"]}

Evaluate the answer against the rubric. For each criterion, decide how well the answer satisfies it (0.0 = not at all, 1.0 = fully), then multiply by the criterion's weight. Sum the weighted scores to get the total (must be between 0.0 and 1.0).

Output ONLY valid JSON:
{{"reasoning": "<evaluate each criterion, give satisfaction * weight, then sum>", "score": <float 0.0-1.0>}}

Example 1 (rubric: Factual Accuracy 0.4, Completeness 0.35, Clarity 0.25):
{{"reasoning": "Factual Accuracy (weight 0.4): answer is fully correct, 1.0 * 0.4 = 0.4. Completeness (weight 0.35): covers main points but misses edge cases, 0.6 * 0.35 = 0.21. Clarity (weight 0.25): well organized and easy to follow, 1.0 * 0.25 = 0.25. Total = 0.86", "score": 0.86}}
Example 2 (rubric: Correctness of Solution 0.5, Use of Examples 0.3, Appropriate Detail 0.2):
{{"reasoning": "Correctness of Solution (weight 0.5): correct approach but has an arithmetic error in the final step, 0.8 * 0.5 = 0.4. Use of Examples (weight 0.3): no examples provided, 0.0 * 0.3 = 0.0. Appropriate Detail (weight 0.2): gives a brief answer without elaboration, 0.2 * 0.2 = 0.04. Total = 0.44", "score": 0.44}}

Your evaluation:""",
        },
    ],
    "judge_binary": _binary_judge_messages,
    "rlcer_verifier": _rlcer_verifier_messages,
    "rrd_initial_rubric_generation": lambda content: [
        {"role": "system", "content": DEFAULT_RRD_RUBRIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Role: You are a rubric designer for an LLM-as-judge system.
Inputs you will receive:
- Prompt: the task/question the response must answer.
- Responses: a set of responses to be evaluated against rubrics.
Goal: Design a comprehensive set of rubrics for evaluating responses to the given prompt.
Only write rubrics you are confident about. Only propose the best new rubrics.
Requirements:
- Propose rubrics that collectively cover the most important dimensions needed to judge whether a response correctly and helpfully satisfies the prompt.
- Each rubric must be consistently judgeable across many responses (avoid vague wording like "good", "nice", "high-quality").
- Each rubric must be prompt-specific (tied to what the user asked), not generic writing advice.
- Each rubric should be written as a single criterion with clear, binary pass/fail boundaries. Prefer objective checks.
- New rubric MUST NOT answer the question directly.
- New rubric MUST NOT repeat any of the responses provided.
{RRD_RUBRIC_TIPS_BLOCK}
Below are the inputs:
- Here is the user prompt for which we want to generate a rubric:
PROMPT: {content["question"]}
- Here are a list of reference responses:
RESPONSES:
{_format_rrd_responses(content["responses"])}
Output STRICTLY in below format. No other text is allowed:
- <RUBRIC> [specific prompt-tied criterion with clear binary pass/fail boundary] </RUBRIC>
- <RUBRIC> [another specific, non-overlapping criterion] </RUBRIC>
Do NOT output literal placeholders like "Rubric 1", "Rubric 2", or "New rubric 1".""",
        },
    ],
    "rrd_rubric_decomposition": lambda content: [
        {"role": "system", "content": DEFAULT_RRD_RUBRIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Role: You are a rubric designer for an LLM-as-judge system.
Inputs you will receive:
- Prompt: the task/question the response must answer.
- Responses: a set of responses to be evaluated against rubrics.
- Current rubric: criterion currently used by a judge. This rubric has already been satisfied by multiple responses, so it is too coarse and fails to distinguish response quality.
- Other rubrics: other rubrics that the new rubric must NOT overlap with.
Goal: Propose exactly TWO new rubrics that are more granular than the existing ones and can better differentiate candidate responses.
Only write rubrics you are confident about. Only propose the best new rubrics.
What "more granular" means (requirements):
- Each new rubric must target a specific, discriminative dimension of quality that is not sufficiently captured by the existing rubrics (e.g., completeness of key sub-steps, correctness of constraints, justification quality, handling of edge cases, faithfulness to prompt format).
- New rubrics should NOT miss critical information contained in the existing rubric.
- Each rubric must be consistently judgeable across many responses (avoid vague wording like "good", "nice", "high-quality").
- Each rubric must be prompt-specific (tied to what the user asked), not generic writing advice.
- Each rubric should be written as a single criterion with clear, binary pass/fail boundaries. Prefer objective checks.
- New rubric MUST NOT repeat any of the responses provided.
- New rubric MUST NOT answer the question directly.
Tips for writing good rubrics:
- The same tips as initial rubric generation apply.
Below are the inputs:
- Here is the user prompt for which we want to generate a rubric:
PROMPT: {content["question"]}
- Here are a list of reference responses:
RESPONSES:
{_format_rrd_responses(content["responses"])}
- Here is existing rubric to improve:
RUBRIC: {content["current_rubric"]}
- Here is other rubric that the new rubric should NOT overlap with:
OTHER RUBRIC:
{_format_rrd_rubrics(content["other_rubrics"])}
Output STRICTLY in below format. No other text is allowed:
- <RUBRIC> [first specific decomposed criterion with clear binary boundary] </RUBRIC>
- <RUBRIC> [second specific decomposed criterion with clear binary boundary] </RUBRIC>
Do NOT output literal placeholders like "Rubric 1", "Rubric 2", or "New rubric 1".""",
        },
    ],
    "rrd_overlap_check": lambda content: [
        {"role": "system", "content": DEFAULT_RRD_FILTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""You are checking whether a new rubric substantially overlaps with ANY of the existing rubrics.
If ANY overlap is found, output YES; otherwise output NO.
Definition of substantial overlapping:
- The new rubric has the same intent as an existing rubric, or is a strict subset/superset of it, or at least 70% of its meaning is covered by an existing rubric so applying both would not materially change scoring outcomes.
- Match on meaning, not wording. Treat synonyms, paraphrases, and inverses with the same effect as overlapping (e.g., "be concise" and "avoid unnecessary verbosity").
- Ignore trivial phrasing, tone, and example differences unless they change the requirement.
Edge cases:
- If scopes are disjoint (different capability or goal), output NO.
- If the new rubric adds only minor qualifiers (e.g., "clearly" or "appropriately") without changing evaluation, output YES.
- If the new rubric merely narrows context while keeping the same criterion (subset) or broadens it (superset), output YES.
Input format:
- EXISTING_RUBRICS:
{_format_rrd_rubrics(content["existing_rubrics"])}
- NEW_RUBRIC: {content["new_rubric"]}
Output STRICTLY in below format. No other text is allowed:
- <EVALUATION> YES/NO </EVALUATION>""",
        },
    ],
    "rrd_conflict_check": lambda content: [
        {"role": "system", "content": DEFAULT_RRD_FILTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""You are checking whether a new rubric expresses opposite meaning of ANY of the existing rubrics.
If ANY opposition is found, output YES; otherwise output NO.
Definition of opposition:
- "Opposite" means the new rubric asserts the negation or reverse polarity of the same requirement, property, or direction as an existing rubric.
- Examples:
  require X vs forbid or avoid X
  must include X vs must NOT include X
  prefer more of X vs prefer less of X (same X, opposite direction)
  answer should be optimistic vs answer should be pessimistic
- Do NOT flag different targets or contexts.
- Do NOT flag orthogonal dimensions (e.g., tone vs citations or "be clear" vs "be concise").
- Do NOT flag mere differences in emphasis, strength, scope, or style.
- Do NOT flag stricter or looser thresholds unless they clearly reverse direction on the same axis.
Input format:
- EXISTING_RUBRICS:
{_format_rrd_rubrics(content["existing_rubrics"])}
- NEW_RUBRIC: {content["new_rubric"]}
Output STRICTLY in below format. No other text is allowed:
- <EVALUATION> YES/NO </EVALUATION>""",
        },
    ],
    "rrd_weight_assignment": lambda content: [
        {"role": "system", "content": DEFAULT_RRD_WEIGHT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Assign a non-negative importance weight for each rubric.
Higher weight means higher importance in pairwise preference judging.

QUESTION:
{content["question"]}

RUBRICS:
{chr(10).join([f"{i + 1}. {r}" for i, r in enumerate(content["rubrics"])])}

Output JSON only:
{{
  "weights": [w1, w2, ..., wN]
}}""",
        },
    ],
    # =========================================================================
    # RaR paper baselines and methods (Gunjal et al., 2025)
    # =========================================================================
    "direct_likert": lambda content: [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator. Given a user prompt and a generated response, "
                "please rate the overall quality of the response on a scale of 1 to 10, where "
                "1 is very poor and 10 is excellent.\n"
                'Start your response with a valid JSON object that starts with "```json" and ends '
                'with "```". The JSON object should contain a single key "rating" and the value '
                "should be an integer between 1 and 10.\n"
                "Example response:\n"
                "```json\n"
                '{\n  "rating": 8\n}\n'
                "```"
            ),
        },
        {
            "role": "user",
            "content": (
                "Given the following prompt, and response, please rate the overall quality "
                "of the response on a scale of 1 to 10.\n\n"
                f"<prompt>\n{content['question']}\n</prompt>\n"
                f"<response>\n{content['answer']}\n</response>\n\n"
                "Your JSON Evaluation:"
            ),
        },
    ],
    "reference_likert": lambda content: [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator. Given a user prompt, a reference response, and a "
                "generated response, please rate the overall quality of the generated response on "
                "a scale of 1 to 10 based on how well it compares to the reference response.\n"
                "Consider factors such as accuracy, completeness, coherence, and helpfulness when "
                "comparing to the reference. The reference response represents a high-quality answer "
                "that you should use as a benchmark.\n"
                'Start your response with a valid JSON object that starts with "```json" and ends '
                'with "```". The JSON object should contain a single key "rating" and the value '
                "should be an integer between 1 and 10.\n"
                "Example response:\n"
                "```json\n"
                '{\n  "rating": 8\n}\n'
                "```"
            ),
        },
        {
            "role": "user",
            "content": (
                "Given the following prompt, reference response, and generated response, please "
                "rate the overall quality of the generated response on a scale of 1 to 10 based "
                "on how well it compares to the reference.\n\n"
                f"<prompt>\n{content['question']}\n</prompt>\n"
                f"<reference_response>\n{content['reference_answer']}\n</reference_response>\n"
                f"<generated_response>\n{content['answer']}\n</generated_response>\n\n"
                "Your JSON Evaluation:"
            ),
        },
    ],
    "rar_implicit": lambda content: [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator. Given a user prompt, a generated response, and a "
                "list of quality rubrics, please rate the overall quality of the response on a "
                "scale of 1 to 10 based on how well it satisfies the rubrics.\n"
                "Consider all rubrics holistically when determining your score. A response that "
                "violates multiple rubrics should receive a lower score, while a response that "
                "satisfies all rubrics should receive a higher score.\n"
                'Start your response with a valid JSON object that starts with "```json" and ends '
                'with "```". The JSON object should contain a single key "rating" and the value '
                "should be an integer between 1 and 10.\n"
                "Example response:\n"
                "```json\n"
                '{\n  "rating": 7\n}\n'
                "```"
            ),
        },
        {
            "role": "user",
            "content": (
                "Given the following prompt, response, and rubrics, please rate the overall "
                "quality of the response on a scale of 1 to 10 based on how well it satisfies "
                "the rubrics.\n\n"
                f"<prompt>\n{content['question']}\n</prompt>\n"
                f"<response>\n{content['answer']}\n</response>\n"
                f"<rubrics>\n{content['rubric']}\n</rubrics>\n\n"
                "Your JSON Evaluation:"
            ),
        },
    ],
    "rar_rubric_generation": lambda content: [
        {
            "role": "user",
            "content": f"""You are an expert rubric writer. Your job is to generate a self-contained set of evaluation criteria ("rubrics") for judging how good a response is to a given question.

Rubrics can cover aspects such as factual correctness, instruction following, reasoning quality, completeness, clarity, helpfulness, safety, style, and common pitfalls. Each rubric item must be self-contained so that a judge can apply it without relying on external knowledge beyond the prompt and the rubric item itself.

Inputs:
- question: The full question text.
- sample_responses: Example candidate responses for this prompt. These are only grounding examples of likely strengths and failure modes. Do NOT copy them, and do NOT let them over-constrain the rubric.

Question:
{content["question"]}

Sample responses:
{_format_rrd_responses(content["responses"])}

Total items:
- Choose 7-20 rubric items depending on prompt complexity.

Each rubric item must include exactly three keys:
1. title (2-4 words)
2. description: One sentence beginning with its category prefix, explicitly stating what to look for.
3. weight: For Essential/Important/Optional, use 1-5 (5 = most important). For Pitfall, use -1 or -2.

Allowed category prefixes:
- Essential Criteria:
- Important Criteria:
- Optional Criteria:
- Pitfall Criteria:

Category guidance:
- Essential: Critical requirements, core facts, or major instruction-following constraints. Missing these should strongly hurt quality.
- Important: Key reasoning, completeness, clarity, or usefulness factors that strongly affect answer quality.
- Optional: Helpful style, extra depth, or polish that improves the response but is not strictly required.
- Pitfall: Common mistakes, omissions, unsafe advice, or instruction violations that should count against the response.

Format notes:
- Make the rubric prompt-specific, but general enough to apply to multiple different valid answers.
- Each item must evaluate one distinct aspect only.
- Avoid domain-specific assumptions unless the question itself requires them.
- Do not copy large blocks from the question or sample responses.
- Prefer criteria that can be judged from the response itself.

Output:
Return a JSON array of rubric objects. Each object must contain exactly three keys: title, description, and weight. Do not include any extra text before or after the JSON array.""",
        }
    ],
    "baseline": lambda content: [
        {"role": "system", "content": content.get("system_prompt", DEFAULT_BASELINE_SYSTEM_PROMPT)},
        {
            "role": "user",
            "content": f"""Question: {content["question"]}

Provide a straightforward answer.""",
        },
    ],
    "question_inference": lambda content: [
        {"role": "system", "content": DEFAULT_QUESTION_INFERENCE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Here is an answer that was written in response to some question:

{content["answer"]}

What question was this answer trying to respond to? Output only the inferred question.""",
        },
    ],
    "policy_with_rubric": lambda content: [
        {
            "role": "system",
            "content": f"""{DEFAULT_POLICY_SYSTEM_PROMPT}

When answering, follow this rubric to ensure a high-quality response:

{content["rubric"]}""",
        },
        {"role": "user", "content": content["question"]},
    ],
    "rlcer_rubric_generation": lambda content: [
        {
            "role": "system",
            "content": (
                "You are an expert in educational assessment and rubric design. "
                "Your task is to analyze a given question-answer pair and generate comprehensive "
                "evaluation rubrics that can be used to assess response quality for this question. "
                "The answer is only a reference answer from the student (not necessarily a good "
                "response), so your rubric system should consider the merits already presented in "
                "the response, and most importantly, room for further improvements. "
                "Output valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""# Input Data

[Question]:
{content["question"]}

[Response]:
{content["response"]}

# Task Instructions

Based on the provided question and answer, generate a comprehensive rubric that's suitable with multiple evaluation criteria. Each criterion should be:
1. Specific and Measurable: Clearly define what constitutes meeting or not meeting the criterion
2. Binary Evaluable: Can be assessed as true/false by an LLM evaluator
3. Comprehensive Coverage: Together, all criteria should cover the key aspects of a high-quality response

# Required Rubric Categories

Generate criteria to cover these aspects:
1. Effectiveness of Problem Decomposition & Planning
Definition: The systematic breakdown of a complex problem into logically ordered, manageable sub-tasks, and the formulation of a strategic roadmap for solving them.
Core Aspects: Identifying key components, establishing step dependencies, sequencing operations, and anticipating challenges before execution.

2. Effectiveness of BackTracking / Self-Validation / Error Handling
Definition: The continuous monitoring of reasoning validity through consistency checks, proactive error detection, and adaptive revision of flawed steps.
Core Aspects: Implementing verification checkpoints (e.g., unit analysis/estimation), diagnosing inconsistencies, recovering from dead ends via path correction (not restarting), and mitigating error propagation.

3. Reasoning Clarity & Flow
Definition: The coherent structuring and communication of logical progression where each step explicitly follows from prior conclusions and leads to subsequent insights.
Core Aspects: Hierarchical argument organization under clear objective, unambiguous terminology, explicit logical connectors (e.g., "thus," "since," "conversely"), and seamless transitions between ideas.

4. Reasoning Focus & Efficiency
Definition: The maintenance of strict alignment with problem objectives while optimizing cognitive resources through elimination of redundancies and irrelevant explorations.
Core Aspects: Sustained direction toward core goals, pruning of tangential paths, and avoidance of overthinking/repetition during reasoning.

5. Other Question-Specific Aspects
Definition: The strategic selection and application of domain-relevant methods that align with both problem constraints and solution goals.
Core Aspects: Leveraging context-optimal approaches (e.g., dimensional analysis in physics, elimination methods for multiple-choice), avoiding technique mismatches, and adapting tools to exploit problem structure (e.g., symmetry in geometry).

# Output Format

Return a JSON object with this structure:
```json
{{
  "question_domain": "math/calculus",
  "rubrics": [
    {{
      "category": "Effectiveness of Problem Decomposition & Planning",
      "criterion": "clear, specific evaluation criterion text",
      "points": <integer score>
    }},
    ...
  ],
  "maximum_score": <sum of all positive points>,
  "minimum_score": <sum of all negative points>,
  "current_score": <score assigned to the reference response>
}}
```

# Important Guidelines

- Output 5-15 criteria in total, ensuring comprehensive coverage.
- Points should reflect the relative importance of each criterion.
- Positive points (1 to 5) should reward merits in the reasoning process.
- Negative points (-5 to -1) should penalize flaws in the reasoning process.
- Try to make your rubrics specific to the question, but not so detailed as some of the overly detailed metrics may not apply to all the response.
- Do not let the reference answer constrain your rubric system.
- The reference response should score strictly below the average of the proposed rubric system so the rubric leaves room for improvement.
- Return only the JSON object without additional commentary.""",
        },
    ],
    "rubric_arm_rubric_generation": lambda content: [
        {
            "role": "user",
            "content": f"""Your task is to extract a set of rubric-style instructions from a user's request.
These rubrics will be used as evaluation criteria to check if a response fully meets the request.
Every rubric item must be a universal principle. If any rubric still contains topic-specific references (e.g., names, places, myths, numbers, historical facts), it is automatically invalid.

- **Two Distinct Categories:**
  - [Hard Rule]: Derived strictly from explicit requirements stated in the request (format, length, structure, forbidden/required elements, etc.).
  - [Principle]: Derived by abstracting any concrete cues into domain-agnostic quality criteria (e.g., clarity, correctness, sound reasoning, pedagogy).
- **Comprehensiveness:** The rubric must cover all critical aspects implied by the request, including explicit requirements and implicit quality standards.
- **Conciseness & Uniqueness:** Each rubric must capture a distinct evaluation criterion. Overlapping or redundant criteria must be merged. Wording must be precise and free of repetition.
- **Format Requirements:**
  - Use a numbered list.
  - Each item starts with "The response" phrased in third person.
  - Append [Hard Rule] or [Principle] at the end of each item.
  - Do not include reasoning, explanations, or examples—only the rubrics.

Here is the request:
{content["question"]}

Please generate the rubrics for the above request.""",
        }
    ],
    "rubric_arm_judge_pairwise": lambda content: [
        {
            "role": "user",
            "content": f"""You are a fair and impartial judge. Your task is to evaluate Response A and Response B based on a given instruction and a rubric. You will conduct this evaluation in distinct phases as outlined below.

### Phase 1: Compliance Check Instructions
First, identify the single most important, objective Gatekeeper Criterion from the rubric.
- A rule is objective (and likely a Gatekeeper) if it can be verified without opinion. Key examples are: word/paragraph limits, required output format (e.g., JSON validity), required/forbidden sections, or forbidden content.
- Conversely, a rule is subjective if it requires interpretation or qualitative judgment. Subjective rules about quality are NOT Gatekeepers.

### Phase 2: Analyze Each Response
Next, for each Gatekeeper Criterion and all other criteria in the rubric, evaluate each response item by item.

### Phase 3: Final Judgment Instructions
Based on the results from the previous phases, determine the winner. Provide a final justification explaining your decision first and then give your decision.

---

### REQUIRED OUTPUT FORMAT
You must follow this exact output format below.

--- Compliance Check ---
Identified Gatekeeper Criterion: <e.g., Criterion 1: Must be under 50 words.>

--- Analysis ---
**Response A:**
- Criterion 1: Justification: <...>
- Criterion 2: Justification: <...>
(and so on for all criteria)

**Response B:**
- Criterion 1: Justification: <...>
- Criterion 2: Justification: <...>
(and so on for all criteria)

--- Final Judgment ---
Justification: <...>
Winner: <Response A / Response B>

---

Task to Evaluate:

Instruction:
{content["instruction"]}

Rubric:
{content["rubric"]}

Response A:
{content["response_a"]}

Response B:
{content["response_b"]}""",
        }
    ],
    "judge_likert_per_item": lambda content: [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator. Rate how well a response conforms to a specific "
                "criterion on a scale of 1-10. 1 = completely fails the criterion, "
                "10 = perfectly satisfies the criterion. Output valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"<RESPONSE>{content['answer']}</RESPONSE>\n\n"
                f"<CRITERION>{content['rubric']}</CRITERION>\n\n"
                "Rate the response's conformity to this criterion on a scale of 1-10.\n"
                'Output STRICTLY as: {"rating": <integer 1-10>}'
            ),
        },
    ],
}


def format_messages(
    message_type: str,
    content: dict[str, Any],
    tokenizer: PreTrainedTokenizer | None = None,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    return_messages: bool = False,
) -> list[dict[str, str]] | list[int]:
    """
    Format messages using the registry-based system.

    Args:
        message_type: Type of message to format. Must be one of: "policy", "rubric_generation", "rubric_generation_v0", "rubric_generation_v3", "rubric_generation_with_examples", "judge", "judge_v1", "judge_binary", "rlcer_verifier", "rrd_initial_rubric_generation", "rrd_rubric_decomposition", "rrd_overlap_check", "rrd_conflict_check", "rrd_weight_assignment", "direct_likert", "reference_likert", "rar_implicit", "rar_rubric_generation", "baseline", "question_inference", "policy_with_rubric", "rlcer_rubric_generation", "judge_likert_per_item", "rubric_arm_rubric_generation", "rubric_arm_judge_pairwise"
        content: Dictionary containing the content for the message type:
            - "policy": {"question": str}
            - "rubric_generation": {"question": str}
            - "rubric_generation_v0": {"question": str}  (pre-2026-03-04, kept for reproducibility)
            - "rubric_generation_correctness": {"question": str}
            - "rubric_generation_with_examples": {"question": str, "policy_answer": str, "baseline_answer": str}
            - "judge": {"question": str, "rubric": str, "answer": str}
            - "judge_v1": {"question": str, "rubric": str, "answer": str}  (pre-2026-03-04, kept for reproducibility)
            - "judge_binary": {"question": str, "rubric": str, "answer": str}
            - "rlcer_verifier": {"question": str, "answer": str, "rubrics": list[dict[str, Any]]}
            - "rrd_initial_rubric_generation": {"question": str, "responses": list[str]}
            - "rrd_rubric_decomposition": {"question": str, "responses": list[str], "current_rubric": str, "other_rubrics": list[str]}
            - "rrd_overlap_check": {"existing_rubrics": list[str], "new_rubric": str}
            - "rrd_conflict_check": {"existing_rubrics": list[str], "new_rubric": str}
            - "rrd_weight_assignment": {"question": str, "rubrics": list[str]}
            - "direct_likert": {"question": str, "answer": str}
            - "reference_likert": {"question": str, "answer": str, "reference_answer": str}
            - "rar_implicit": {"question": str, "answer": str, "rubric": str}
            - "rar_rubric_generation": {"question": str, "responses": list[str]}
            - "baseline": {"question": str, "system_prompt": Optional[str]} (system_prompt defaults to DEFAULT_BASELINE_SYSTEM_PROMPT)
            - "question_inference": {"answer": str}
            - "policy_with_rubric": {"question": str, "rubric": str}
            - "rlcer_rubric_generation": {"question": str, "response": str}
            - "judge_likert_per_item": {"answer": str, "rubric": str}
            - "rubric_arm_rubric_generation": {"question": str}
            - "rubric_arm_judge_pairwise": {"instruction": str, "rubric": str, "response_a": str, "response_b": str}
        tokenizer: Tokenizer to use for tokenization. If provided and tokenize=True, messages will be tokenized.
        tokenize: Whether to tokenize the messages. Defaults to True. Requires tokenizer to be provided.
        add_generation_prompt: Whether to add generation prompt when tokenizing (e.g., "<|assistant|>\n")

    Returns:
        If tokenize=False: List of message dictionaries with 'role' and 'content' keys
        If tokenize=True: List[int] of token IDs (with pad tokens removed)

    Raises:
        ValueError: If message_type is not in registry or if tokenize=True but tokenizer is None
    """
    if message_type not in MESSAGE_FORMATTER_REGISTRY:
        raise ValueError(
            f"Unknown message_type '{message_type}'. Available types: {list(MESSAGE_FORMATTER_REGISTRY.keys())}"
        )

    if tokenize and tokenizer is None:
        raise ValueError("tokenizer must be provided when tokenize=True")

    # Get formatter from registry
    formatter = MESSAGE_FORMATTER_REGISTRY[message_type]

    # Format messages
    messages = formatter(content)

    # Tokenize if requested
    if tokenize:
        if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
            raise ValueError("Tokenizer does not support chat template")

        token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True)

        # Remove pad tokens if present
        if tokenizer.pad_token_id is not None:
            token_ids = [x for x in token_ids if x != tokenizer.pad_token_id]

        if return_messages:
            return token_ids, messages
        else:
            return token_ids

    return messages


# Mapping from rubric prompt keys to their system prompt constants.
# Used by training scripts to resolve --rubric_prompt_key into the actual
# system prompt string for dataset tokenization and inference consistency.
RUBRIC_PROMPT_KEY_TO_SYSTEM_PROMPT: dict[str, str] = {
    "rubric_generation": DEFAULT_RUBRIC_GENERATION_SYSTEM_PROMPT,
    "rubric_generation_v0": V0_RUBRIC_GENERATION_SYSTEM_PROMPT,
    "rubric_generation_v3": V3_RUBRIC_GENERATION_SYSTEM_PROMPT,
    "rubric_generation_correctness": CORRECTNESS_FOCUSED_RUBRIC_GENERATION_SYSTEM_PROMPT,
}


def get_rubric_system_prompt(prompt_key: str) -> str:
    """Get the rubric generation system prompt for a given registry key.

    Args:
        prompt_key: A key from MESSAGE_FORMATTER_REGISTRY that corresponds to a
            rubric generation prompt (e.g. "rubric_generation", "rubric_generation_v3").

    Returns:
        The system prompt string.

    Raises:
        ValueError: If the prompt_key is not a known rubric generation key.
    """
    if prompt_key not in RUBRIC_PROMPT_KEY_TO_SYSTEM_PROMPT:
        raise ValueError(
            f"Unknown rubric prompt key '{prompt_key}'. "
            f"Available: {list(RUBRIC_PROMPT_KEY_TO_SYSTEM_PROMPT.keys())}"
        )
    return RUBRIC_PROMPT_KEY_TO_SYSTEM_PROMPT[prompt_key]
