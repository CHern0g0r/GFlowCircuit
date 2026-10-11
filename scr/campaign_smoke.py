"""Exercise the real two-circuit supervisor and separate mapping with tiny budgets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from src.circuit_artifacts import circuit_metrics, sha256, write_json
from src.diversity_campaign import DEFAULT_PROTOCOL, load_protocol
from src.sampling_diversity import _abc_quote, _resolve_abc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--abc-path", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--methods", nargs="+", default=["reinforce", "drills", "ppo", "gflownet_tb"])
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    abc = _resolve_abc(args.abc_path)
    protocol = load_protocol(DEFAULT_PROTOCOL)
    circuits = ["C1355", "C5315"]
    summary = {}
    for method in args.methods:
        command = [sys.executable, "-m", "src.diversity_campaign", "--smoke", "run-method",
                   "--method", method, "--workers", "2", "--circuits", *circuits,
                   "--artifact-root", str(root), "--abc-path", str(abc), "--device", args.device]
        for stage in ("train-sample", "mapping"):
            subprocess.run([*command, "--stage", stage], check=True)
            # Repeat to exercise validated reuse without creating new attempts.
            subprocess.run([*command, "--stage", stage], check=True)
            saved = json.loads((root / "methods" / method / "status.json").read_text())
            assert saved["complete"] and all(r["state"] == "skipped" for r in saved["circuits"].values())
            if stage == "train-sample":
                assert not any(root.glob(f"{method}_*/attempt_*/diversity"))
        for circuit in circuits:
            attempts = list((root / f"{method}_{circuit}").glob("attempt_*"))
            assert len(attempts) == 1
            attempt = attempts[0]
            source = Path(__file__).resolve().parents[1] / protocol["baseline"]["circuits"][circuit]["circuit_path"]
            records = json.loads((attempt / "diversity/mapping_records.json").read_text())
            checked = set()
            for row in records:
                assert circuit_metrics(Path(row["aig_path"]), 2) == (row["aig_size"], row["aig_depth"])
                for name in ("aig_path", "lut_path"):
                    path = Path(row[name])
                    assert path.is_relative_to(attempt)
                    if path in checked:
                        continue
                    result = subprocess.run([str(abc), "-c", f"cec -n {_abc_quote(source)} {_abc_quote(path)}"],
                                            capture_output=True, text=True, timeout=120)
                    assert result.returncode == 0 and "Networks are equivalent" in result.stdout, result.stdout
                    checked.add(path)
            def rows(path):
                with path.open() as stream:
                    return list(csv.DictReader(stream))
            full = rows(attempt / "samples_seed_0.csv")
            for budget in (2, 4):
                assert rows(attempt / f"samples_seed_0_n{budget}.csv") == [r for r in full if int(r["sample_id"]) < budget]
            metrics = json.loads((attempt / "diversity/metrics.json").read_text())
            for group in metrics["groups"]:
                for quality in group["metrics"]["quality"].values():
                    for winner in quality.values():
                        if winner and winner["selected_artifact"]:
                            path = Path(winner["selected_artifact"])
                            assert path.is_relative_to(attempt) and sha256(path) == winner["selected_sha256"]
            summary[f"{method}/{circuit}"] = {"complete": True, "equivalence_checks": len(checked)}
        write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
