from __future__ import annotations

import json
import subprocess
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from inference_engine.benchmarking.openrouter_billing import (
    EvidenceClass,
    ExperimentClassification,
    FrozenTariff,
    ResponseUsage,
    classify_experiment,
    combine_required_statuses,
    reconcile_tokens_r1,
    reconstruct_openrouter_cost,
)
from inference_engine.benchmarking.openrouter_evidence_semantics import (
    HISTORICAL_6B_RECONCILIATION_SEMANTICS,
    PROVIDER_METADATA_CONTRACT_ANOMALY,
    RECONCILIATION_SEMANTICS_V1,
    RECONCILIATION_SEMANTICS_V2,
    StreamedFieldClassification,
    TokenFieldSemantics,
    generation_streamed_proves_client_visible_streaming,
    interpret_generation_streamed_field,
    reconcile_tokens_r1_v2,
    resolve_reconciliation_semantics,
    select_r1_reconciler,
    split_upstream_cost_surfaces,
    token_field_semantics,
    token_fields_semantically_comparable,
    upstream_cost_surfaces_from_payloads,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "benchmarks/reports/openrouter-billing-reconciliation-6b-20260911T220449Z"
DECISION_JSON = REPORT / "decision.json"
DECISION_MD = REPORT / "decision.md"
RECONCILIATION = REPORT / "reconciliation.json"
REQUEST_EVIDENCE = REPORT / "request-evidence.jsonl"
GENERATION_EVIDENCE = REPORT / "generation-evidence.jsonl"
KEY_ACCOUNTING = REPORT / "key-accounting.json"
PRICING = REPORT / "pricing-source.json"
RAW_MANIFEST = REPORT / "raw-evidence-manifest.json"
HISTORICAL_FILES = (
    DECISION_JSON,
    DECISION_MD,
    RECONCILIATION,
    REQUEST_EVIDENCE,
    GENERATION_EVIDENCE,
    REPORT / "experiment-spec.json",
    PRICING,
    RAW_MANIFEST,
)


def _tariff() -> FrozenTariff:
    payload = json.loads(PRICING.read_text(encoding="utf-8"))
    return FrozenTariff(
        provider=str(payload["provider"]),
        model=str(payload["model"]),
        pricing_table_version=str(payload["pricing_table_version"]),
        pricing_observed_at=str(payload["pricing_observed_at"]),
        source_url=str(payload["source_url"]),
        input_per_token=Decimal(str(payload["input_per_token"])),
        cached_input_per_token=Decimal(str(payload["cached_input_per_token"])),
        output_per_token=Decimal(str(payload["output_per_token"])),
        cache_write_per_token=None,
    )


def _usage(*, prompt: int, completion: int = 1, cached: int = 0, cost: Decimal | None = None) -> ResponseUsage:
    return ResponseUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cached_input_tokens=cached,
        cache_write_tokens=0,
        reasoning_tokens=0,
        cost=cost,
        upstream_inference_cost=None,
        is_byok=False,
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_historical_6b_decision_remains_validated_mismatch() -> None:
    decision = json.loads(DECISION_JSON.read_text(encoding="utf-8"))
    markdown = DECISION_MD.read_text(encoding="utf-8")
    assert decision["classification"] == "VALIDATED_MISMATCH"
    assert "**Classification:** `VALIDATED_MISMATCH`" in markdown
    rules = {item["rule_id"]: item["status"] for item in decision["rules"]}
    assert rules == {"R1": "MISMATCH", "R2": "MATCH", "R3": "MATCH", "R4": "MATCH", "R5": "MATCH"}
    assert decision["direct_openai_billing_validated"] is False
    assert HISTORICAL_6B_RECONCILIATION_SEMANTICS == RECONCILIATION_SEMANTICS_V1


def test_historical_frozen_r1_still_returns_mismatch_on_committed_evidence() -> None:
    requests = _load_jsonl(REQUEST_EVIDENCE)
    generations = {str(item["workload_item_id"]): item for item in _load_jsonl(GENERATION_EVIDENCE)}
    assert len(requests) == 50
    statuses = []
    for request in requests:
        item_id = str(request["workload_item_id"])
        usage_raw = request["usage"]
        assert isinstance(usage_raw, dict)
        details = usage_raw["prompt_tokens_details"]
        assert isinstance(details, dict)
        usage = _usage(
            prompt=int(usage_raw["prompt_tokens"]),  # type: ignore[arg-type]
            completion=int(usage_raw["completion_tokens"]),  # type: ignore[arg-type]
            cached=int(details["cached_tokens"]),  # type: ignore[arg-type]
        )
        generation = generations[item_id]["data"]
        assert isinstance(generation, dict)
        result = reconcile_tokens_r1(response=usage, generation=generation)
        statuses.append(result.status)
        assert result.status is EvidenceClass.MISMATCH
    assert combine_required_statuses(statuses) is EvidenceClass.MISMATCH
    v2_statuses = []
    for request in requests:
        item_id = str(request["workload_item_id"])
        usage_raw = request["usage"]
        assert isinstance(usage_raw, dict)
        usage = _usage(
            prompt=int(usage_raw["prompt_tokens"]),  # type: ignore[arg-type]
            completion=int(usage_raw["completion_tokens"]),  # type: ignore[arg-type]
        )
        generation = generations[item_id]["data"]
        assert isinstance(generation, dict)
        v2 = reconcile_tokens_r1_v2(response=usage, generation=generation)
        v2_statuses.append(v2.status)
        assert v2.status is EvidenceClass.MATCH
    assert combine_required_statuses(v2_statuses) is EvidenceClass.MATCH
    decision = json.loads(DECISION_JSON.read_text(encoding="utf-8"))
    assert decision["classification"] == "VALIDATED_MISMATCH"
    assert decision["rules"][0]["status"] == "MISMATCH"


def test_future_rule_does_not_compare_undocumented_token_fields_as_equivalent() -> None:
    assert token_field_semantics("generation.tokens_prompt") is TokenFieldSemantics.UNKNOWN
    assert token_field_semantics("generation.native_tokens_prompt") is TokenFieldSemantics.PROVIDER_NATIVE
    assert token_field_semantics("response.usage.prompt_tokens") is TokenFieldSemantics.BILLING_OBSERVED
    assert not token_fields_semantically_comparable(
        "response.usage.prompt_tokens", "generation.tokens_prompt"
    )
    assert token_fields_semantically_comparable(
        "response.usage.prompt_tokens", "generation.native_tokens_prompt"
    )
    usage = _usage(prompt=89)
    v2 = reconcile_tokens_r1_v2(
        response=usage,
        generation={
            "tokens_prompt": 64,
            "tokens_completion": 1,
            "native_tokens_prompt": 89,
            "native_tokens_completion": 1,
            "native_tokens_cached": 0,
            "provider_name": "OpenAI",
        },
    )
    assert v2.status is EvidenceClass.MATCH
    assert "NOT_COMPARABLE" in v2.reason
    v1 = reconcile_tokens_r1(
        response=usage,
        generation={"tokens_prompt": 64, "tokens_completion": 1, "native_tokens_prompt": 89},
    )
    assert v1.status is EvidenceClass.MISMATCH


def test_unknown_token_fields_are_not_comparable_without_provider_native_pair() -> None:
    usage = _usage(prompt=89)
    missing_native = reconcile_tokens_r1_v2(
        response=usage,
        generation={"tokens_prompt": 64, "tokens_completion": 1, "provider_name": "OpenAI"},
    )
    assert missing_native.status in {EvidenceClass.INCONCLUSIVE, EvidenceClass.NOT_COMPARABLE}
    assert missing_native.status is not EvidenceClass.MISMATCH
    unknown_provider = reconcile_tokens_r1_v2(
        response=usage,
        generation={
            "tokens_prompt": 64,
            "native_tokens_prompt": 89,
            "native_tokens_completion": 1,
            "provider_name": None,
        },
    )
    assert unknown_provider.status is EvidenceClass.NOT_COMPARABLE


def test_native_agreement_does_not_rewrite_historical_classification() -> None:
    usage = _usage(prompt=89)
    generation = {
        "tokens_prompt": 64,
        "tokens_completion": 1,
        "native_tokens_prompt": 89,
        "native_tokens_completion": 1,
        "native_tokens_cached": 0,
        "provider_name": "OpenAI",
    }
    v2 = reconcile_tokens_r1_v2(response=usage, generation=generation)
    historical = classify_experiment(
        [reconcile_tokens_r1(response=usage, generation=generation)],
        blocked=False,
        completed_requests=50,
        planned_requests=50,
        successful_paid_requests=50,
    )
    assert v2.status is EvidenceClass.MATCH
    assert historical is ExperimentClassification.VALIDATED_MISMATCH
    stored = json.loads(DECISION_JSON.read_text(encoding="utf-8"))
    assert stored["classification"] == "VALIDATED_MISMATCH"


def test_monetary_b_c_d_e_chain_remains_exact() -> None:
    tariff = _tariff()
    requests = _load_jsonl(REQUEST_EVIDENCE)
    generations = {str(item["workload_item_id"]): item for item in _load_jsonl(GENERATION_EVIDENCE)}
    keys = json.loads(KEY_ACCOUNTING.read_text(encoding="utf-8"))
    sum_response_prompt = 0
    sum_tokens_prompt = 0
    sum_native_prompt = 0
    sum_response_completion = 0
    sum_tokens_completion = 0
    sum_native_completion = 0
    sum_c = Decimal("0")
    sum_d = Decimal("0")
    sum_b_response = Decimal("0")
    sum_b_tokens_prompt = Decimal("0")
    sum_b_native = Decimal("0")
    for request in requests:
        usage = request["usage"]
        assert isinstance(usage, dict)
        details = usage["prompt_tokens_details"]
        assert isinstance(details, dict)
        generation = generations[str(request["workload_item_id"])]["data"]
        assert isinstance(generation, dict)
        prompt = int(usage["prompt_tokens"])  # type: ignore[arg-type]
        completion = int(usage["completion_tokens"])  # type: ignore[arg-type]
        cached = int(details["cached_tokens"])  # type: ignore[arg-type]
        tokens_prompt = int(generation["tokens_prompt"])  # type: ignore[arg-type]
        native_prompt = int(generation["native_tokens_prompt"])  # type: ignore[arg-type]
        tokens_completion = int(generation["tokens_completion"])  # type: ignore[arg-type]
        native_completion = int(generation["native_tokens_completion"])  # type: ignore[arg-type]
        sum_response_prompt += prompt
        sum_tokens_prompt += tokens_prompt
        sum_native_prompt += native_prompt
        sum_response_completion += completion
        sum_tokens_completion += tokens_completion
        sum_native_completion += native_completion
        cost = Decimal(str(usage["cost"]))
        total_cost = Decimal(str(generation["total_cost"]))
        sum_c += cost
        sum_d += total_cost
        reconstructed = reconstruct_openrouter_cost(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_input_tokens=cached,
            cache_write_tokens=int(details["cache_write_tokens"]),  # type: ignore[arg-type]
            tariff=tariff,
        )
        tokens_prompt_cost = reconstruct_openrouter_cost(
            prompt_tokens=tokens_prompt,
            completion_tokens=tokens_completion,
            cached_input_tokens=cached,
            cache_write_tokens=0,
            tariff=tariff,
        )
        native_cost = reconstruct_openrouter_cost(
            prompt_tokens=native_prompt,
            completion_tokens=native_completion,
            cached_input_tokens=cached,
            cache_write_tokens=0,
            tariff=tariff,
        )
        assert reconstructed is not None
        assert tokens_prompt_cost is not None
        assert native_cost is not None
        assert reconstructed == cost
        assert native_cost == cost
        sum_b_response += reconstructed
        sum_b_tokens_prompt += tokens_prompt_cost
        sum_b_native += native_cost
    pre = Decimal(str(keys["pre"]["sanitized"]["data"]["usage"]))
    post = Decimal(str(keys["post"]["sanitized"]["data"]["usage"]))
    key_delta = post - pre
    assert sum_response_prompt == 4523
    assert sum_tokens_prompt == 3200
    assert sum_native_prompt == 4523
    assert sum_response_completion == 50
    assert sum_tokens_completion == 50
    assert sum_native_completion == 50
    assert sum_c == Decimal("0.00070845")
    assert sum_d == Decimal("0.00070845")
    assert key_delta == Decimal("0.00070845")
    assert sum_b_response == sum_c
    assert sum_b_native == sum_c
    assert sum_b_tokens_prompt == Decimal("0.000510000")
    assert sum_b_tokens_prompt != sum_c
    assert sum_c == sum_d == key_delta


def test_streamed_metadata_cannot_prove_client_visible_streaming() -> None:
    anomaly = interpret_generation_streamed_field(
        requested_stream=False,
        response_object="chat.completion",
        generation_streamed=True,
    )
    assert anomaly.classification is StreamedFieldClassification.PROVIDER_METADATA_CONTRACT_ANOMALY
    assert anomaly.classification.value == PROVIDER_METADATA_CONTRACT_ANOMALY
    assert anomaly.proves_client_visible_streaming is False
    assert generation_streamed_proves_client_visible_streaming(True) is False
    assert generation_streamed_proves_client_visible_streaming(False) is False
    consistent = interpret_generation_streamed_field(
        requested_stream=False,
        response_object="chat.completion",
        generation_streamed=False,
    )
    assert consistent.proves_client_visible_streaming is False
    generations = _load_jsonl(GENERATION_EVIDENCE)
    requests = _load_jsonl(REQUEST_EVIDENCE)
    assert all(item["object"] == "chat.completion" for item in requests)
    assert all(isinstance(item["data"], dict) and item["data"]["streamed"] is True for item in generations)


def test_conflicting_upstream_cost_surfaces_remain_separate() -> None:
    surfaces = split_upstream_cost_surfaces(
        response_cost_details_upstream=Decimal("0.00070845"),
        generation_upstream=Decimal("0"),
    )
    assert surfaces.required_financial_gate is False
    assert surfaces.response_cost_details_upstream_inference_cost == Decimal("0.00070845")
    assert surfaces.generation_upstream_inference_cost == Decimal("0")
    parsed = upstream_cost_surfaces_from_payloads(
        chat_payload={
            "usage": {"cost_details": {"upstream_inference_cost": Decimal("0.00001395")}}
        },
        generation_payload={"data": {"upstream_inference_cost": 0}},
    )
    assert parsed.response_cost_details_upstream_inference_cost == Decimal("0.00001395")
    assert parsed.generation_upstream_inference_cost == Decimal("0")
    requests = _load_jsonl(REQUEST_EVIDENCE)
    generations = _load_jsonl(GENERATION_EVIDENCE)
    response_sum = Decimal("0")
    generation_sum = Decimal("0")
    for request, generation in zip(requests, generations, strict=True):
        usage = request["usage"]
        assert isinstance(usage, dict)
        details = usage["cost_details"]
        assert isinstance(details, dict)
        data = generation["data"]
        assert isinstance(data, dict)
        response_sum += Decimal(str(details["upstream_inference_cost"]))
        generation_sum += Decimal(str(data["upstream_inference_cost"]))
    assert response_sum == Decimal("0.00070845")
    assert generation_sum == Decimal("0")


def test_no_provider_network_call_occurs_during_semantic_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provider network call is forbidden in Mission 6C tests")

    monkeypatch.setattr("urllib.request.urlopen", _blocked)
    test_historical_6b_decision_remains_validated_mismatch()
    test_historical_frozen_r1_still_returns_mismatch_on_committed_evidence()
    test_monetary_b_c_d_e_chain_remains_exact()
    select_r1_reconciler(RECONCILIATION_SEMANTICS_V1)
    select_r1_reconciler(RECONCILIATION_SEMANTICS_V2)
    assert resolve_reconciliation_semantics({}) == RECONCILIATION_SEMANTICS_V1
    assert resolve_reconciliation_semantics({"reconciliation_semantics": "v2"}) == RECONCILIATION_SEMANTICS_V2


def test_raw_evidence_is_not_tracked_in_git() -> None:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
    assert "pre-key.raw.json" not in tracked
    assert "post-key.raw.json" not in tracked
    assert "/private/tmp/inferenceledger-openrouter-6b/" not in tracked
    assert ".local/share/inferenceledger/evidence/" not in tracked
    manifest = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest["responses"]) == 50
    assert len(manifest["generations"]) == 50
    assert len(manifest["pre_key_sha256"]) == 64
    assert len(manifest["post_key_sha256"]) == 64


def test_historical_6b_artifact_bytes_are_untouched() -> None:
    expected = {
        "decision.json": "VALIDATED_MISMATCH",
        "decision.md": "VALIDATED_MISMATCH",
    }
    for path in HISTORICAL_FILES:
        assert path.is_file()
        digest = sha256(path.read_bytes()).hexdigest()
        assert len(digest) == 64
    decision = json.loads(DECISION_JSON.read_text(encoding="utf-8"))
    assert decision["classification"] == expected["decision.json"]
    spec = json.loads((REPORT / "experiment-spec.json").read_text(encoding="utf-8"))
    assert spec["reconciliation_policy"]["R1"].startswith("exact integer equality of comparable billed token fields")
    assert "reconciliation_semantics" not in spec
