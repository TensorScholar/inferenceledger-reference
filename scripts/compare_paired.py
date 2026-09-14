from __future__ import annotations

import argparse

from inference_engine.benchmarking.context_store import SQLiteBenchmarkContextStore
from inference_engine.benchmarking.paired_comparison import (
    compare_paired_runs,
    write_paired_run_evidence,
)
from inference_engine.benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from inference_engine.benchmarking.statistics import PairedBootstrapConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="compare_paired",
        description="Build workload-paired statistical evidence from two stored benchmark runs.",
    )
    parser.add_argument("--sqlite-ledger-path", default="reports/benchmarks/ledger.sqlite3")
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument(
        "--output-path",
        default="reports/benchmarks/latest-paired-comparison.json",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--minimum-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return _run(parser.parse_args())


def _run(args: argparse.Namespace) -> int:
    from pathlib import Path

    ledger_path = Path(args.sqlite_ledger_path)
    ledger = SQLiteBenchmarkLedger(ledger_path)
    contexts = SQLiteBenchmarkContextStore(ledger_path)
    evidence = compare_paired_runs(
        baseline_run_id=args.baseline_run_id,
        candidate_run_id=args.candidate_run_id,
        baseline_report=ledger.get_report(args.baseline_run_id),
        candidate_report=ledger.get_report(args.candidate_run_id),
        baseline_contexts=contexts.get_contexts(args.baseline_run_id),
        candidate_contexts=contexts.get_contexts(args.candidate_run_id),
        baseline_traces=ledger.get_traces(args.baseline_run_id),
        candidate_traces=ledger.get_traces(args.candidate_run_id),
        bootstrap_config=PairedBootstrapConfig(
            confidence_level=args.confidence_level,
            bootstrap_iterations=args.bootstrap_iterations,
            minimum_samples=args.minimum_samples,
            seed=args.seed,
        ),
    )
    output_path = Path(args.output_path)
    write_paired_run_evidence(evidence, output_path)
    print(
        " ".join(
            [
                f"available={str(evidence.available).lower()}",
                f"baseline_run_id={evidence.baseline_run_id}",
                f"candidate_run_id={evidence.candidate_run_id}",
                f"workload_items={evidence.workload_item_count}",
                f"segments={len(evidence.segments)}",
                f"output_path={output_path}",
            ]
        )
    )
    return 0 if evidence.available else 1


if __name__ == "__main__":
    raise SystemExit(main())
