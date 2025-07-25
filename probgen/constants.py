import os

from pathlib import Path

FEASIBILITY_DEFINITION = "Feasibility is the likelihood that the claim can be reproduced based on current scientific knowledge and current technology given appropriate resources."
FEASIBILITY_SCORE_DEFINITIONS = {
    -2: "Extremely unlikely to be feasible. Significant doubts. 95% confident it's infeasible.",
    -1: "Somewhat unlikely to be feasible. Moderate doubts against but cannot rule out.",
    0: "Neither unlikely nor likely to be feasible. Not enough data, no strong argument for or against.",
    1: "Somewhat likely to be feasible. Moderate doubts for it but it might be possible.",
    2: "Extremely likely to be feasible. Minor to no doubts. 95% confident it's feasible.",
}

FEASIBILITY_SCORE_DEFINITIONS_STR = "\n".join(
    [
        f"{score}: {definition}"
        for score, definition in sorted(FEASIBILITY_SCORE_DEFINITIONS.items())
    ]
)

GOLD_STANDARD_PATH = os.path.join(
    Path(__file__).parent.parent, "scify-problems", "gold-standard", "materials"
)
GOLD_STANDARD_ALLOYS_PATH = Path(os.path.join(GOLD_STANDARD_PATH, "alloys"))
GOLD_STANDARD_BATTERIES_PATH = Path(os.path.join(GOLD_STANDARD_PATH, "batteries"))
GOLD_STANDARD_SEMICONDUCTORS_PATH = Path(
    os.path.join(GOLD_STANDARD_PATH, "semiconductors")
)
GOLD_STANDARD_SUPERCONDUCTORS_PATH = Path(
    os.path.join(GOLD_STANDARD_PATH, "superconductors")
)
