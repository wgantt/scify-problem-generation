import argparse
import json
import random

from pathlib import Path
from scify_formats.formats import GoldStandard
from typing import Optional

SEED = 14607
random.seed(SEED)  # Set a seed for reproducibility

def postprocess(
    input_file: Path,
    output_dir: Path,
    problem_id_prefix: str,
    subdomain: str,
    domain: str = "materials",
    author: str = "JHU",
    comment: Optional[str] = None,
) -> None:
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    for item in data:
        # shuffle to ensure problem number doesn't
        # correlate with feasibility score
        responses = item["response"]
        random.shuffle(responses)
        for i, r in enumerate(responses):
            problem_id = f"{problem_id_prefix}-{i + 1}"
            gs = GoldStandard(
                type="gold standard",
                format_version="1.0",
                problem_id=problem_id,
                problem_version="1.0",
                domain=domain,
                subdomain=subdomain,
                claim=r["claim"],
                artifacts=[],  # no artifacts supported for now
                likert_score=r["likert_score"],
                explanation=r["explanation"],
                evidence={}, # no evidence supported for now
                author=author,
                comments=[comment] if comment else [],
            )
            with open(
                output_dir / f"{problem_id}.jsonl", "w", encoding="utf-8"
            ) as out_file:
                out_file.write(gs.model_dump_json() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Postprocess SciFy feasibility problems"
    )
    parser.add_argument(
        "input_file", type=Path, help="Input JSONL file with feasibility problems"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory to save postprocessed files"
    )
    parser.add_argument("output_file_prefix", type=str, help="Prefix for output files")
    parser.add_argument("subdomain", type=str, help="Subdomain for the problems")
    parser.add_argument(
        "--domain",
        type=str,
        default="materials",
        help="Domain of the problems (default: materials)",
    )
    parser.add_argument(
        "--author",
        type=str,
        default="JHU",
        help="Author of the problems (default: JHU)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    postprocess(
        args.input_file,
        args.output_dir,
        args.output_file_prefix,
        args.subdomain,
        args.domain,
        args.author,
    )
