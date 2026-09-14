from __future__ import annotations

import argparse
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

from inference_engine.benchmarking.experiment_provenance import (
    require_committed_clean_experiment_spec,
)
from inference_engine.benchmarking.openrouter_billing import (
    WORKLOAD_ITEM_COUNT,
    UrllibTransport,
    env_key_presence,
    load_json_preserving_decimals,
    require_experiment_id,
    require_openrouter_gateway,
    require_openrouter_model,
    workload_sha256_hex,
)
from inference_engine.benchmarking.openrouter_billing_run import (
    run_openrouter_billing_experiment,
    validate_experiment_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-register or execute Mission 6B-OR OpenRouter gateway billing reconciliation."
    )
    parser.add_argument(
        "--experiment-path",
        default=(
            "benchmarks/experiments/"
            "openrouter-billing-reconciliation-6b-20260911T220449Z/experiment.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/reports/openrouter-billing-reconciliation-6b-20260911T220449Z",
    )
    parser.add_argument(
        "--raw-dir",
        default="/private/tmp/inferenceledger-openrouter-6b/openrouter-billing-reconciliation-6b-20260911T220449Z",
    )
    parser.add_argument(
        "--key-preflight",
        action="store_true",
        help="Read-only GET /api/v1/key after Git-proven pre-registration. No inference.",
    )
    parser.add_argument(
        "--execute-paid",
        action="store_true",
        help="Run the frozen paid workload once. Default is protocol validation only.",
    )
    args = parser.parse_args()
    if args.execute_paid and args.key_preflight:
        raise SystemExit("choose either --key-preflight or --execute-paid, not both")
    repo_root = Path(__file__).resolve().parents[1]
    spec_path = Path(args.experiment_path)
    if not spec_path.is_absolute():
        spec_path = repo_root / spec_path
    spec = _load_object(spec_path)
    validate_experiment_spec(spec)
    require_experiment_id(str(spec["experiment_id"]))
    runtime = spec["runtime"]
    require_openrouter_gateway(str(runtime["execution_gateway"]))
    require_openrouter_model(str(runtime["requested_model"]))
    spend = spec["spend_policy"]
    if int(spend["planned_request_count"]) != WORKLOAD_ITEM_COUNT:
        raise SystemExit("committed Mission 6B-OR spec must freeze 50 requests")
    workload_path = Path(str(spec["workload"]["path"]))
    if not workload_path.is_absolute():
        workload_path = repo_root / workload_path
    actual = sha256(workload_path.read_bytes()).hexdigest()
    expected = str(spec["workload"]["sha256"])
    if actual != expected:
        raise SystemExit(f"workload SHA-256 mismatch expected={expected} actual={actual}")
    generated = workload_sha256_hex(str(spec["experiment_id"]))
    if generated != expected:
        raise SystemExit("committed workload does not match the frozen generator")
    snapshot = require_committed_clean_experiment_spec(spec_path)
    print(
        " ".join(
            [
                f"experiment_id={spec['experiment_id']}",
                f"spec_sha256={snapshot.spec_sha256}",
                f"source_commit={snapshot.source_commit_sha}",
                f"OPENROUTER_API_KEY={env_key_presence()}",
                f"execute_paid={args.execute_paid}",
                f"key_preflight={args.key_preflight}",
            ]
        )
    )
    if not args.execute_paid and not args.key_preflight:
        return 0
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    raw_dir = Path(args.raw_dir)
    result = run_openrouter_billing_experiment(
        spec_path=spec_path,
        output_dir=output_dir,
        raw_dir=raw_dir,
        repo_root=repo_root,
        transport=UrllibTransport(),
        env=os.environ,
        execute_paid=bool(args.execute_paid),
        key_preflight_only=bool(args.key_preflight),
    )
    print(
        " ".join(
            [
                f"classification={result.get('classification')}",
                f"completed={result.get('completed_request_count')}",
                f"successes={result.get('successful_request_count')}",
                f"retries={result.get('retry_count')}",
                f"response_cost_sum={result.get('response_cost_sum')}",
                f"key_usage_delta={result.get('key_usage_delta')}",
            ]
        )
    )
    if result.get("user_action"):
        print(f"user_action={result['user_action']}")
    return 0 if result.get("classification") != "BLOCKED" else 2


def _load_object(path: Path) -> dict[str, Any]:
    parsed = load_json_preserving_decimals(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
