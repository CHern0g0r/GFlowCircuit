#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.zhu2020_analysis import write_analysis_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Zhu 2020 reproduction analysis artifacts.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="GFlowCircuit repository root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for report and CSV outputs. Default: <repo-root>/reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else repo_root / "reports"
    paths = write_analysis_outputs(repo_root=repo_root, output_root=output_root)
    print("Generated Zhu 2020 reproduction analysis:")
    for label, path in paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
