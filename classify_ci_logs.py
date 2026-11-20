"""Classify CI build logs by likely failure cause.

The published ci-builds-repair file contains failed builds and log text, but no
human-written failure-category label. This script therefore creates weak labels
from explicit log signatures and keeps the matching evidence for review.

Usage:
    python classify_ci_logs.py
    python classify_ci_logs.py --input hf://datasets/.../test-00000-of-00001.parquet

Install dependencies if needed:
    pip install pandas pyarrow huggingface_hub
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# Ordered from specific to broad. A row receives the first matching category.
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    (
        "dependency_installation",
        [
            r"no matching distribution",
            r"could not find a version that satisfies",
            r"pip install",
            r"error: subprocess-exited-with-error",
            r"failed building wheel",
            r"resolutionimpossible",
        ],
    ),
    (
        "import_or_module_error",
        [
            r"modulenotfounderror",
            r"importerror",
            r"cannot import name",
            r"no module named",
        ],
    ),
    (
        "test_failure",
        [
            r"pytest",
            r"unittest",
            r"assertionerror",
            r"failed tests?",
            r"\btest_\w+",
            r"tests? failed",
        ],
    ),
    (
        "coverage_failure",
        [
            r"coverage report",
            r"fail-under",
            r"coverage.*below",
            r"coverage.*threshold",
        ],
    ),
    (
        "lint_or_formatting",
        [
            r"ruff",
            r"flake8",
            r"pylint",
            r"black",
            r"isort",
            r"yapf",
            r"eslint",
            r"prettier",
            r"format(?:ting)? check",
            r"lint(?:ing)? check",
        ],
    ),
    (
        "type_checking",
        [r"mypy", r"pyright", r"typescript", r"type error", r"typecheck"],
    ),
    (
        "build_or_packaging",
        [
            r"setup\.py",
            r"pyproject\.toml",
            r"building wheel",
            r"package(?:d|ing)",
            r"failed to build",
            r"cmake",
            r"compilation",
        ],
    ),
    (
        "environment_or_resource",
        [
            r"out of memory",
            r"oom",
            r"permission denied",
            r"timed out",
            r"timeout",
            r"killed",
            r"cuda",
            r"docker",
            r"runner",
        ],
    ),
]


def flatten_logs(value: object) -> str:
    """Turn the dataset's list-of-dicts log field into searchable text."""
    # PyArrow may materialize nested Parquet lists as numpy arrays.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        value = value.tolist()
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict):
                step = str(item.get("step_name", ""))
                log = str(item.get("log", ""))
                chunks.append(f"[{step}]\n{log}")
            else:
                chunks.append(str(item))
        return "\n\n".join(chunks)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def classify_log(text: str) -> tuple[str, str]:
    """Return (category, matched signatures) for one log."""
    lowered = text.lower()
    for category, patterns in CATEGORY_RULES:
        matches = [pattern for pattern in patterns if re.search(pattern, lowered)]
        if matches:
            return category, "; ".join(matches)
    return "unknown", ""


def classify_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["log_text"] = result["logs"].map(flatten_logs)
    classified = result["log_text"].map(classify_log)
    result[["failure_category", "matched_signatures"]] = pd.DataFrame(
        classified.tolist(), index=result.index
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="hf://datasets/icmlbenchname/ci-builds-repair/data/test-00000-of-00001.parquet",
        help="Local Parquet path or an hf:// URI.",
    )
    parser.add_argument(
        "--output",
        default="classified_ci_builds.parquet",
        help="Output Parquet path (a CSV with the same stem is also written).",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if "logs" not in df.columns:
        raise ValueError("Expected a 'logs' column in the input dataset")

    classified = classify_frame(df)
    output_path = Path(args.output)
    classified.to_parquet(output_path, index=False)
    classified.drop(columns=["logs"], errors="ignore").to_csv(
        output_path.with_suffix(".csv"), index=False
    )

    print(f"Loaded {len(df):,} rows")
    print("Failure categories:")
    print(classified["failure_category"].value_counts(dropna=False).to_string())
    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
