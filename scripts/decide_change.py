from __future__ import annotations

import argparse
from pathlib import Path

from inference_engine.application.services.benchmarking import (
    MigrationDecisionRequest,
    decide_stored_change,
)
from inference_engine.benchmarking.pilot_contract import SemanticEnforcementMode


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="decide_change",
        description=(
            "Pair two stored runs, evaluate the pre-registered Change Gate, and write the "
            "authoritative migration decision bundle."
        ),
    )
    parser.add_argument("--experiment-path", required=True)
    parser.add_argument("--sqlite-ledger-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-json", default=None)
    parser.add_argument(
        "--semantic-mode",
        choices=[mode.value for mode in SemanticEnforcementMode],
        default=SemanticEnforcementMode.FULL_INFERENCELEDGER.value,
        help=(
            "FULL_INFERENCELEDGER enforces pilot semantic validity at the study-decision boundary; "
            "SEMANTICS_ABLATION uses the same evidence and Change Gate but bypasses only that override."
        ),
    )
    return _run(parser.parse_args())


def _run(args: argparse.Namespace) -> int:
    semantic_mode = getattr(
        args,
        "semantic_mode",
        SemanticEnforcementMode.FULL_INFERENCELEDGER.value,
    )
    result = decide_stored_change(
        MigrationDecisionRequest(
            experiment_path=Path(args.experiment_path),
            sqlite_ledger_path=Path(args.sqlite_ledger_path),
            output_dir=Path(args.output_dir),
            runtime_json_path=Path(args.runtime_json) if args.runtime_json else None,
            semantic_mode=SemanticEnforcementMode(semantic_mode),
        )
    )
    print(result.message)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
