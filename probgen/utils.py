import json

from glob import glob
from pathlib import Path
from typing import Any, Dict, List


def load_gold_standard_problem_from_file(
    problem_file: Path, jsonl: bool = False
) -> List[Dict[str, Any]]:
    """
    Load a gold standard problem from a JSON file.

    Args:
        problem_file (Path): Path to the JSON file containing the problem data.
        jsonl (bool): If True, treats the file as a JSONL file. Defaults to False.

    Returns:
        list: The loaded problem data.
    """
    with open(problem_file, "r") as f:
        problem_data = []
        if jsonl:
            for line in f:
                problem_data.append(json.loads(line))
        else:
            problem_data.append(json.load(f))
    return problem_data


def load_gold_standard_problems_from_dir(
    directory: Path, jsonl: bool = False
) -> List[Dict[str, Any]]:
    """
    Load all gold standard problems from a specified directory.

    Args:
        directory (Path): Directory containing the gold standard problem JSON files.
        jsonl (bool): If True, treats the files as JSONL files. Defaults to False.

    Returns:
        list: A list of dictionaries, each representing a gold standard problem.
    """
    if jsonl:
        problem_files = glob(str(directory / "*.jsonl"))
    else:
        problem_files = glob(str(directory / "*.json"))
    problems = []
    for file in problem_files:
        problems.extend(load_gold_standard_problem_from_file(Path(file), jsonl=jsonl))
    return problems


def load_gold_standard_problems(
    problems_path: Path, jsonl: bool = False
) -> List[Dict[str, Any]]:
    """
    Load gold standard problems from a specified path.

    Args:
        problems_path (Path): Path to the directory containing gold standard problems.
        jsonl (bool): If True, treats the path as a JSONL file. Defaults to False.

    Returns:
        list: A list of dictionaries, each representing a gold standard problem.
    """
    if problems_path.is_file():
        return load_gold_standard_problem_from_file(problems_path, jsonl=jsonl)
    elif problems_path.is_dir():
        return load_gold_standard_problems_from_dir(problems_path, jsonl=jsonl)
    else:
        raise ValueError(f"Invalid path: {problems_path}. Must be a file or directory.")
