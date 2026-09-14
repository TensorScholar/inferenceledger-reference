from __future__ import annotations

import argparse
from pathlib import Path

from inference_engine.application.services.benchmarking.litellm_arm_service import (
    LiteLLMArmExecutionRequest,
    _completion_kwargs,
    _generation_settings,
    _quality_limitation,
    execute_litellm_arm,
)

__all__ = [
    "_completion_kwargs",
    "_generation_settings",
    "_quality_limitation",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_litellm_workload",
        description="Execute one pre-registered experiment arm through LiteLLM and persist evidence.",
    )
    parser.add_argument("--experiment-path", required=True)
    parser.add_argument("--arm", required=True, choices=["baseline", "candidate"])
    parser.add_argument("--sqlite-ledger-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--raw-dir",
        default="/tmp/inferenceledger-external-raw",
        help="Directory for unsanitized StandardLoggingPayload files. Must stay outside git.",
    )
    return _run(parser.parse_args())


def _run(args: argparse.Namespace) -> int:
    result = execute_litellm_arm(
        LiteLLMArmExecutionRequest(
            experiment_path=Path(args.experiment_path),
            arm=str(args.arm),
            sqlite_ledger_path=Path(args.sqlite_ledger_path),
            output_dir=Path(args.output_dir),
            raw_dir=Path(args.raw_dir),
        )
    )
    print(result.message)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
