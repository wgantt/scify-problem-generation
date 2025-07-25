from pathlib import Path
from typing import Any, Dict, List

from probgen.constants import FEASIBILITY_DEFINITION, FEASIBILITY_SCORE_DEFINITIONS_STR
from probgen.utils import load_gold_standard_problems


MODIFY_FEASIBILITY_SYSTEM_PROMPT_V1 = f"""You are a world-renowned researcher in materials science. I will provide you with the following information:
- Claim: A scientific claim describing some result in materials science.
- Context: An artifact that provides context for the claim, such as a press release.
- Feasibility Score: A score from -2 to 2 indicating the feasibility of the claim.
- Explanation: A scientifically grounded justification for the feasibility score.

Here is the definition of FEASIBILITY: {FEASIBILITY_DEFINITION}.

Here are the definitions of the feasibility scores:
{FEASIBILITY_SCORE_DEFINITIONS_STR}

Given this information, your task is to provide four minimal modifications to the original claim, one for each possible feasibility score OTHER THAN the original one, along with a modified explanation for the new feasibility score. You do not have to provide a modified context.The modified explanation for each modified claim must be scientifically accurate and must adequately account for why the modified claim has the modified feasibility score that it does. Each modified claim should also be as similar as possible to the original claim EXCEPT for its feasibility. The modified explanation must also be similar in style to the original explanation. Please respond with a JSON object containing the following fields:
- "claim": The modified claim.
- "likert_score": The modified feasibility score.
- "explanation": The modified explanation for the feasibility score.
DO NOT include any additional text in your response, just the JSON object.
"""

MODIFY_FEASIBILITY_USER_PROMPT_V1 = "Claim: {claim}\nContext: {artifact}\nFeasibility Score: {score}\nExplanation: {explanation}"


def format_modify_feasibility_user_prompt(problem: Dict[str, Any]) -> str:
    """
    Formats the user prompt for modifying feasibility.

    Args:
        claim (str): The original scientific claim.
        artifact (str): The context artifact related to the claim.
        score (int): The feasibility score of the claim.
        explanation (str): The explanation for the feasibility score.

    Returns:
        str: Formatted user prompt.
    """
    assert (
        len(problem["artifacts"]) == 1
    ), "Expected exactly one artifact in the problem."
    claim = problem["claim"]
    artifact = problem["artifacts"][0]["text"]
    score = problem["likert_score"]
    explanation = " ".join([s["text"] for s in problem["explanation"]])
    return MODIFY_FEASIBILITY_USER_PROMPT_V1.format(
        claim=claim, artifact=artifact, score=score, explanation=explanation
    )


def construct_modify_feasibility_prompt(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs a modify feasibility example from the given problem.

    Args:
        problem (Dict[str, Any]): The problem containing claim, artifacts, score, and explanation.

    Returns:
        Dict[str, Any]: A dictionary containing the system prompt, user prompt, and metadata.
    """
    return {
        "system_prompt": MODIFY_FEASIBILITY_SYSTEM_PROMPT_V1,
        "user_prompt": format_modify_feasibility_user_prompt(problem),
        "meta": {"problem": problem},
    }


def construct_modify_feasibility_prompts(problems_path: Path) -> List[Dict[str, Any]]:
    """
    Constructs modify feasibility prompts for all problems in the specified path.

    Args:
        problems_path (Path): Path to the directory containing gold standard problems.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing a system prompt, user prompt, and metadata.
    """
    from probgen.utils import load_gold_standard_problems

    problems = load_gold_standard_problems(problems_path)
    return [construct_modify_feasibility_prompt(problem) for problem in problems]
