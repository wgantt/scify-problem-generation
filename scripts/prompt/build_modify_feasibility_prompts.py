import os
import json

from pathlib import Path

from probgen.constants import (
    GOLD_STANDARD_ALLOYS_PATH,
    GOLD_STANDARD_BATTERIES_PATH,
    GOLD_STANDARD_SEMICONDUCTORS_PATH,
    GOLD_STANDARD_SUPERCONDUCTORS_PATH,
)
from probgen.prompt.modify_feasibility import construct_modify_feasibility_prompts

OUTPUT_ROOT = Path("prompts/modify-feasibility")


def main():
    for subdomain, problems_path in {
        "alloys": GOLD_STANDARD_ALLOYS_PATH,
        "batteries": GOLD_STANDARD_BATTERIES_PATH,
        "semiconductors": GOLD_STANDARD_SEMICONDUCTORS_PATH,
        "superconductors": GOLD_STANDARD_SUPERCONDUCTORS_PATH,
    }.items():
        prompts = construct_modify_feasibility_prompts(problems_path)
        for p in prompts:
            problem_id = p["meta"]["problem"]["problem_id"]
            output_path = Path(
                os.path.join(OUTPUT_ROOT, subdomain, f"{problem_id}.jsonl")
            )
            os.makedirs(output_path.parent, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
