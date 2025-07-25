import argparse
import os
import json

from pathlib import Path

from probgen.prompt.verify_claim_and_explanation import (
    construct_verify_claim_and_explanation_prompts,
)

OUTPUT_ROOT = Path("prompts/verify-claim-and-explanation")


def main(input_path: Path, subdomain: str, jsonl: bool) -> None:
    prompts = construct_verify_claim_and_explanation_prompts(input_path, jsonl=jsonl)
    for p in prompts:
        problem_id = p["meta"]["problem"]["problem_id"]
        output_path = Path(os.path.join(OUTPUT_ROOT, subdomain, f"{problem_id}.jsonl"))
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build verify claim and explanation prompts."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the directory containing gold standard problems.",
    )
    parser.add_argument(
        "subdomain",
        type=str,
        choices=["alloys", "batteries", "semiconductors", "superconductors"],
        help="Subdomain for which to build prompts.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="If set, treats the input files as JSONL files instead of JSON.",
    )
    args = parser.parse_args()
    main(args.input_path, args.subdomain, args.jsonl)
