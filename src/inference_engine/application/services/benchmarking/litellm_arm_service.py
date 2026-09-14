from __future__ import annotations

import copy
import importlib
import json
import time
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from ....benchmarking.eval import evaluate_text
from ....benchmarking.experiment_provenance import (
    ExperimentProvenanceError,
    build_execution_provenance,
    execution_interval_from_standard_logging_payloads,
    require_committed_clean_experiment_spec,
)
from ....benchmarking.harness import load_workload
from ....benchmarking.pilot_contract import PilotInstanceContract, load_optional_pilot_contract
from ....benchmarking.segmentation import BenchmarkRequestContext
from ....domain.models.economics import ExactRequestCostEvidence
from ....infrastructure.importers.litellm import (
    import_litellm_standard_logging_chain,
    sanitize_standard_logging_payload,
)
from ....infrastructure.importers.litellm_exact import extract_litellm_exact_cost_evidence
from ....infrastructure.telemetry.request_log import JsonlRequestLog, RequestTrace
from .run_evidence_service import persist_benchmark_run


class _PayloadCapture(Protocol):
    payloads: list[dict[str, object]]


@dataclass(frozen=True)
class LiteLLMArmExecutionRequest:
    """Inputs for the selected external execution/import adapter."""

    experiment_path: Path
    arm: str
    sqlite_ledger_path: Path
    output_dir: Path
    raw_dir: Path


@dataclass(frozen=True)
class LiteLLMArmExecutionResult:
    exit_code: int
    message: str


def execute_litellm_arm(request: LiteLLMArmExecutionRequest) -> LiteLLMArmExecutionResult:
    """Execute one pre-registered LiteLLM arm and cross the canonical run-evidence boundary."""
    if request.arm not in {"baseline", "candidate"}:
        raise ValueError("arm must be baseline or candidate")

    experiment_path = request.experiment_path
    try:
        snapshot = require_committed_clean_experiment_spec(experiment_path)
    except ExperimentProvenanceError as exc:
        raise SystemExit(f"experiment spec provenance: {exc}") from exc

    experiment = _load_json_object(experiment_path)
    pilot_contract = load_optional_pilot_contract(experiment)
    workload_path = Path(str(experiment["workload"]["path"]))
    expected_sha = str(experiment["workload"]["sha256"])
    actual_sha = sha256(workload_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            "workload SHA-256 does not match the pre-registered experiment; "
            f"expected={expected_sha}, actual={actual_sha}"
        )
    _validate_pilot_workload_identity(
        contract=pilot_contract,
        workload_sha256=actual_sha,
        workload_item_count=len(load_workload(workload_path)),
    )

    arm = request.arm
    arm_spec = experiment[arm]
    run_id = str(arm_spec["run_id"])
    runtime_intent = experiment["runtime_intent"]
    model = str(runtime_intent["model_argument"])
    api_base = str(runtime_intent["endpoint"])
    generation = _generation_settings(experiment)
    quality_limitation = _quality_limitation(experiment)
    workload = load_workload(workload_path)
    output_dir = request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = request.raw_dir / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = output_dir / f"{run_id}.jsonl"
    request_log = JsonlRequestLog(ledger_path)
    litellm, capture_logger = _load_litellm()
    capture: _PayloadCapture = capture_logger()
    litellm.logging_callback_manager.add_litellm_success_callback(capture)
    litellm.logging_callback_manager.add_litellm_failure_callback(capture)

    traces: list[RequestTrace] = []
    request_contexts: list[BenchmarkRequestContext] = []
    exact_costs: list[ExactRequestCostEvidence] = []
    sanitized_payloads: list[dict[str, object]] = []
    raw_hashes: list[dict[str, object]] = []

    for item in workload:
        capture.payloads.clear()
        kwargs = _completion_kwargs(
            model=model,
            api_base=api_base,
            prompt=item.prompt,
            generation=generation,
            arm=arm,
            arm_spec=arm_spec if isinstance(arm_spec, dict) else {},
        )
        response_text = ""
        try:
            response = litellm.completion(**kwargs)
            response_text = _completion_text(response)
        except Exception:
            response_text = ""

        payload_chain = _wait_for_payload_chain(capture, workload_item_id=item.id)
        trace = import_litellm_standard_logging_chain(payload_chain)
        if pilot_contract is not None:
            exact_costs.append(
                extract_litellm_exact_cost_evidence(
                    payload_chain,
                    currency=pilot_contract.economic.currency,
                )
            )

        raw_path = raw_dir / f"{item.id}.json"
        raw_bytes = json.dumps(
            list(payload_chain), indent=2, sort_keys=True, default=str
        ).encode("utf-8")
        raw_path.write_bytes(raw_bytes)
        raw_hashes.append(
            {
                "workload_item_id": item.id,
                "logical_path": f"private_raw/{run_id}/{item.id}.json",
                "sha256": sha256(raw_bytes).hexdigest(),
                "byte_size": len(raw_bytes),
                "attempt_count": len(payload_chain),
            }
        )
        sanitized_payloads.extend(
            sanitize_standard_logging_payload(payload) for payload in payload_chain
        )

        eval_result = evaluate_text(response_text, item.eval_spec)
        if eval_result is not None and trace.error_type is None:
            trace = replace(
                trace,
                quality_passed=eval_result.passed,
                quality_score=eval_result.score,
                quality_reason=eval_result.reason,
                eval_type=eval_result.eval_type,
            )
        request_log.append(trace)
        traces.append(trace)
        request_contexts.append(
            BenchmarkRequestContext.from_tags(
                request_id=trace.request_id,
                workload_item_id=item.id,
                tags=item.tags,
            )
        )

    provider = traces[0].provider if traces else "unknown"
    model_name = traces[0].model if traces else "unknown"
    report_path = output_dir / f"{run_id}.report.json"
    segment_path = output_dir / f"{run_id}.segments.json"
    persisted = persist_benchmark_run(
        run_id=run_id,
        workload_path=workload_path,
        strategy=f"litellm_{arm}",
        provider=provider,
        model=model_name,
        ledger_path=ledger_path,
        sqlite_ledger_path=request.sqlite_ledger_path,
        traces=traces,
        request_contexts=request_contexts,
        route_traces=(),
        report_path=report_path,
        segment_report_path=segment_path,
        additional_limitations=(
            "No InferenceLedger route decision was executed; reconciliation is expected to be MISSING_ROUTE.",
            quality_limitation,
        ),
        pilot_acquisition=(
            pilot_contract.acquisition_for_arm(arm) if pilot_contract is not None else None
        ),
        exact_cost_evidence=exact_costs,
    )
    report = persisted.report

    sanitized_path = output_dir / f"{run_id}.sanitized-payloads.jsonl"
    with sanitized_path.open("w", encoding="utf-8") as handle:
        for payload in sanitized_payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    hashes_path = output_dir / f"{run_id}.raw-payload-sha256.json"
    hashes_path.write_text(
        json.dumps(raw_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    interval = execution_interval_from_standard_logging_payloads(sanitized_payloads)
    try:
        provenance = build_execution_provenance(
            snapshot=snapshot,
            interval=interval,
            workload_sha256=actual_sha,
            declared_status=str(experiment["status"]) if experiment.get("status") is not None else None,
            declared_locked_at_utc=(
                str(experiment["locked_at_utc"])
                if experiment.get("locked_at_utc") is not None
                else None
            ),
            require_spec_precedes_execution=True,
        )
    except ExperimentProvenanceError as exc:
        raise SystemExit(f"experiment spec provenance: {exc}") from exc
    provenance_path = output_dir / f"{run_id}.provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    message = " ".join(
        [
            f"run_id={run_id}",
            f"arm={arm}",
            f"requests={report.request_count}",
            f"successes={report.success_count}",
            f"failures={report.failure_count}",
            f"attempts={report.provider_attempt_count}",
            f"retries={report.provider_retry_count}",
            f"quality_pass_rate={report.quality_pass_rate}",
            f"raw_dir={raw_dir}",
            f"report_path={report_path}",
        ]
    )
    return LiteLLMArmExecutionResult(exit_code=0, message=message)


def _validate_pilot_workload_identity(
    *,
    contract: PilotInstanceContract | None,
    workload_sha256: str,
    workload_item_count: int,
) -> None:
    if contract is None:
        return
    if contract.workload.sha256 != workload_sha256:
        raise ValueError("pilot contract workload SHA-256 disagrees with experiment workload")
    if contract.workload.item_count != workload_item_count:
        raise ValueError("pilot contract workload item_count disagrees with frozen workload")


def _generation_settings(experiment: dict[str, Any]) -> dict[str, Any]:
    generation = experiment.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("experiment.generation is required")
    temperature = generation.get("temperature")
    max_tokens = generation.get("max_tokens")
    stream = generation.get("stream", False)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("experiment.generation.temperature must be numeric")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("experiment.generation.max_tokens must be a positive int")
    if not isinstance(stream, bool):
        raise ValueError("experiment.generation.stream must be a boolean")
    if stream:
        raise ValueError("streaming executions are not supported by this runner")
    return {"temperature": float(temperature), "max_tokens": max_tokens, "stream": False}


def _quality_limitation(experiment: dict[str, Any]) -> str:
    evaluator = experiment.get("quality_evaluator")
    if isinstance(evaluator, dict) and isinstance(evaluator.get("measures"), str):
        return f"Quality scoring is deterministic: {evaluator['measures']}"
    return "Quality scoring is deterministic workload-declared evaluation, not semantic quality."


def _completion_kwargs(
    *,
    model: str,
    api_base: str,
    prompt: str,
    generation: dict[str, Any],
    arm: str,
    arm_spec: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "api_base": api_base,
        "temperature": generation["temperature"],
        "max_tokens": generation["max_tokens"],
        "stream": False,
    }
    if arm == "candidate":
        response_format = arm_spec.get("response_format")
        if not isinstance(response_format, dict):
            raise ValueError("candidate.response_format is required for the candidate arm")
        kwargs["response_format"] = response_format
    return kwargs


def _load_litellm() -> tuple[Any, Any]:
    try:
        litellm = importlib.import_module("litellm")
    except ImportError as exc:
        raise SystemExit(
            "LiteLLM is not installed in this interpreter. Use the isolated Mission 4 "
            "environment; do not add LiteLLM to repository dependencies."
        ) from exc

    class StandardLogCapture:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def __call__(self, kwargs: dict[str, Any], *_args: object, **_kwargs: object) -> None:
            payload = kwargs.get("standard_logging_object")
            if payload is None:
                return
            self.payloads.append(_payload_dict(payload))

    return litellm, StandardLogCapture


def _wait_for_payload_chain(
    capture: _PayloadCapture,
    *,
    workload_item_id: str,
    timeout_seconds: float = 15.0,
    settle_seconds: float = 0.5,
) -> tuple[dict[str, object], ...]:
    """Collect the complete LiteLLM callback chain after it becomes quiescent.

    LiteLLM submits logging callbacks asynchronously. Multiple payloads are valid and represent the
    retry/fallback attempt chain; they must not be rejected merely because more than one provider
    invocation occurred.
    """
    deadline = time.monotonic() + timeout_seconds
    last_count = 0
    last_change = time.monotonic()
    while time.monotonic() < deadline:
        count = len(capture.payloads)
        now = time.monotonic()
        if count != last_count:
            last_count = count
            last_change = now
        if count > 0 and now - last_change >= settle_seconds:
            return tuple(copy.deepcopy(capture.payloads))
        time.sleep(0.05)
    raise RuntimeError(
        f"{workload_item_id}: LiteLLM emitted no settled StandardLoggingPayload chain "
        f"within {timeout_seconds:.1f}s"
    )


def _wait_for_single_payload(
    capture: _PayloadCapture,
    *,
    workload_item_id: str,
    timeout_seconds: float = 15.0,
) -> dict[str, object]:
    """Compatibility helper retained for historical unit imports; rejects multi-attempt chains."""
    payloads = _wait_for_payload_chain(
        capture,
        workload_item_id=workload_item_id,
        timeout_seconds=timeout_seconds,
    )
    if len(payloads) != 1:
        raise RuntimeError(
            f"{workload_item_id}: expected exactly one LiteLLM StandardLoggingPayload, "
            f"observed={len(payloads)}"
        )
    return payloads[0]


def _payload_dict(payload: object) -> dict[str, object]:
    if hasattr(payload, "model_dump"):
        dumped = payload.model_dump()
        if isinstance(dumped, dict):
            return {str(key): copy.deepcopy(value) for key, value in dumped.items()}
    if isinstance(payload, dict):
        return {str(key): copy.deepcopy(value) for key, value in payload.items()}
    raise TypeError(f"LiteLLM standard_logging_object has unsupported type {type(payload)!r}")


def _completion_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _load_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw
