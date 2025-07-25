import json

from glob import glob
from pathlib import Path
from typing import Any, Dict, List


def load_gold_standard_problem_from_file(problem_file: Path) -> Dict[str, Any]:
    """
    Load a gold standard problem from a JSON file.

    Args:
        problem_file (Path): Path to the JSON file containing the problem data.

    Returns:
        dict: The loaded problem data.
    """
    with open(problem_file, "r") as f:
        problem_data = json.load(f)
    return problem_data


def load_gold_standard_problems_from_dir(directory: Path) -> List[Dict[str, Any]]:
    """
    Load all gold standard problems from a specified directory.

    Args:
        directory (Path): Directory containing the gold standard problem JSON files.

    Returns:
        list: A list of dictionaries, each representing a gold standard problem.
    """
    problem_files = glob(str(directory / "*.json"))
    problems = [
        load_gold_standard_problem_from_file(Path(file)) for file in problem_files
    ]
    return problems


def load_gold_standard_problems(problems_path: Path) -> List[Dict[str, Any]]:
    """
    Load gold standard problems from a specified path.

    Args:
        problems_path (Path): Path to the directory containing gold standard problems.

    Returns:
        list: A list of dictionaries, each representing a gold standard problem.
    """
    if problems_path.is_file():
        return [load_gold_standard_problem_from_file(problems_path)]
    elif problems_path.is_dir():
        return load_gold_standard_problems_from_dir(problems_path)
    else:
        raise ValueError(f"Invalid path: {problems_path}. Must be a file or directory.")
