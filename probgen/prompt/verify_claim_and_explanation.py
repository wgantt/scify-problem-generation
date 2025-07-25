from pathlib import Path
from typing import Any, Dict, List

from probgen.constants import FEASIBILITY_DEFINITION, FEASIBILITY_SCORE_DEFINITIONS_STR
from probgen.utils import load_gold_standard_problems

VERIFY_CLAIM_AND_EXPLANATION_SYSTEM_PROMPT_V1 = f"""You are a world-renowned researcher in materials science. I will provide you with the following information:
- Claim: A scientific claim describing some result in materials science.
- Feasibility Score: A score from -2 to 2 indicating the feasibility of the claim.
- Explanation: A scientifically grounded justification for the feasibility score.

Here is the definition of FEASIBILITY: {FEASIBILITY_DEFINITION}

Here are the definitions of the possible feasibility scores:
{FEASIBILITY_SCORE_DEFINITIONS_STR}

Given this information, your task is to:
1. Determine whether the feasibility score is correct based on your own background knowledge and knowledge of the problem domain.
2. Determine whether the explanation is scientifically accurate.

Based on your reasoning, you should provide a JSON object containing the following fields:
- "likert_score": The feasibility score YOU think the claim should have, which can be -2, -1, 0, 1, or 2. Note that this may be the same as the original score.
- "explanation": A scientifically accurate explanation for the feasibility score you provided.
"""

VERIFY_CLAIM_AND_EXPLANATION_USER_PROMPT_V1 = (
    "Claim: {claim}\nFeasibility Score: {score}\nExplanation: {explanation}"
)


def format_verify_claim_and_explanation_user_prompt(problem: Dict[str, Any]) -> str:
    """
    Formats the user prompt for verifying claims and explanations.

    Args:
        problem (Dict[str, Any]): The SciFy problem to be verified

    Returns:
        str: Formatted user prompt.
    """
    claim = problem["claim"]
    score = problem["likert_score"]
    if isinstance(problem["explanation"], str):
        explanation = problem["explanation"]
    else:
        explanation = " ".join([s["text"] for s in problem["explanation"]])
    return VERIFY_CLAIM_AND_EXPLANATION_USER_PROMPT_V1.format(
        claim=claim, score=score, explanation=explanation
    )


def construct_verify_claim_and_explanation_prompt(
    problem: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Constructs a verify claim and explanation example from the given problem.

    Args:
        problem (Dict[str, Any]): The problem containing claim, artifacts, score, and explanation.

    Returns:
        Dict[str, Any]: A dictionary containing the system prompt, user prompt, and metadata.
    """
    return {
        "instance_id": problem["problem_id"],
        "system_prompt": VERIFY_CLAIM_AND_EXPLANATION_SYSTEM_PROMPT_V1,
        "user_prompt": format_verify_claim_and_explanation_user_prompt(problem),
        "meta": {"problem": problem},
    }


def construct_verify_claim_and_explanation_prompts(
    problems_path: Path,
    jsonl: bool = False,
) -> List[Dict[str, Any]]:
    """
    Constructs claim and explanation verification prompts for all problems in the specified path.

    Args:
        problems_path (Path): Path to the directory containing gold standard problems.
        jsonl (bool): If True, treats the files as JSONL files. Defaults to False.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing a system prompt, user prompt, and metadata.
    """

    problems = load_gold_standard_problems(problems_path, jsonl=jsonl)
    return [
        construct_verify_claim_and_explanation_prompt(problem) for problem in problems
    ]
