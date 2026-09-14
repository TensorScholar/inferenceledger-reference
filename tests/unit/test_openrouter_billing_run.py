from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from inference_engine.benchmarking.openrouter_billing import (
    REQUESTED_MODEL,
    HttpResponse,
    OpenRouterBillingError,
    conservative_request_cost_usd,
    load_frozen_tariff,
    reconstruct_openrouter_cost,
)
from inference_engine.benchmarking.openrouter_billing_run import (
    PollPolicy,
    issue_chat_completion,
    poll_generation_metadata,
    run_openrouter_billing_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_GIT_TMP = REPO_ROOT / ".tmp-pytest-git"
EXPERIMENT_ID = "openrouter-billing-reconciliation-6b-20260911T000000Z"


@dataclass
class ScriptedTransport:
    responses: list[HttpResponse]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def send(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> HttpResponse:
        assert "Authorization" in headers
        assert "X-OpenRouter-Cache" not in headers
        self.calls.append((method, url))
        if not self.responses:
            raise AssertionError(f"unexpected extra HTTP call {method} {url}")
        return self.responses.pop(0)


@dataclass
class FakeClock:
    now: datetime = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
    sleeps: list[float] = field(default_factory=list)

    def now_utc(self) -> datetime:
        current = self.now
        self.now = self.now + timedelta(milliseconds=25)
        return current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now = self.now + timedelta(seconds=seconds)


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command_env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        **(env or {}),
    }
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.email=or-billing-test@example.com",
            "-c",
            "user.name=OpenRouter Billing Test",
            *args,
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=command_env,
    )
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def _chat_body(
    *,
    cost: str = "0.0001212",
    prompt_tokens: int = 80,
    provider: str = "OpenAI",
    generation_id: str = "gen-test-1",
    response_upstream_cost: str | None = None,
) -> bytes:
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 2,
        "total_tokens": prompt_tokens + 2,
        "cost": cost,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    if response_upstream_cost is not None:
        usage["cost_details"] = {"upstream_inference_cost": response_upstream_cost}
    payload = {
        "id": generation_id,
        "object": "chat.completion",
        "model": REQUESTED_MODEL,
        "provider": provider,
        "usage": usage,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "OK"}}],
    }
    return json.dumps(payload).encode("utf-8")


def _upstream_json_number(value: str | None) -> int | str | None:
    if value is None:
        return None
    if value == "0":
        return 0
    return value


def _generation_body(
    *,
    total_cost: str = "0.0001212",
    prompt_tokens: int = 80,
    tokens_prompt: int | None = None,
    native_tokens_prompt: int | None = None,
    provider_name: str = "OpenAI",
    generation_id: str = "gen-test-1",
    streamed: bool = True,
    upstream_inference_cost: str | None = "0",
) -> bytes:
    payload = {
        "data": {
            "id": generation_id,
            "model": REQUESTED_MODEL,
            "provider_name": provider_name,
            "tokens_prompt": prompt_tokens if tokens_prompt is None else tokens_prompt,
            "tokens_completion": 2,
            "native_tokens_prompt": prompt_tokens if native_tokens_prompt is None else native_tokens_prompt,
            "native_tokens_completion": 2,
            "native_tokens_cached": 0,
            "total_cost": total_cost,
            "upstream_inference_cost": _upstream_json_number(upstream_inference_cost),
            "service_tier": None,
            "finish_reason": "stop",
            "upstream_id": "chatcmpl-test",
            "is_byok": False,
            "streamed": streamed,
        }
    }
    return json.dumps(payload).encode("utf-8")


def _key_body(*, usage: str, limit: str = "0.05", remaining: str = "0.05") -> bytes:
    payload = {
        "data": {
            "usage": usage,
            "limit": limit,
            "limit_remaining": remaining,
            "creator_user_id": "user_should_not_be_committed",
            "label": "sk-or-v1-should-redact",
            "is_management_key": False,
            "is_provisioning_key": False,
        }
    }
    return json.dumps(payload).encode("utf-8")


def test_failed_inference_is_not_retried() -> None:
    transport = ScriptedTransport(
        responses=[
            HttpResponse(status_code=500, body=b'{"error":{"message":"provider"}}'),
            HttpResponse(status_code=200, body=_chat_body()),
        ]
    )
    first = issue_chat_completion(transport, api_key="test-key", prompt="one", max_tokens=8)
    assert first.status_code == 500
    assert len(transport.calls) == 1
    assert transport.calls[0] == ("POST", "https://openrouter.ai/api/v1/chat/completions")


def test_generation_metadata_retries_get_only() -> None:
    transport = ScriptedTransport(
        responses=[
            HttpResponse(status_code=404, body=b'{"error":{"message":"not ready"}}'),
            HttpResponse(status_code=200, body=_generation_body()),
        ]
    )
    clock = FakeClock()
    result = poll_generation_metadata(
        transport,
        api_key="test-key",
        generation_id="gen-test-1",
        policy=PollPolicy(
            generation_max_attempts=3,
            generation_initial_delay_seconds=0.5,
            generation_backoff_multiplier=2.0,
            generation_max_delay_seconds=8.0,
            key_max_attempts=1,
            key_delay_seconds=1.0,
            key_stable_reads=3,
        ),
        clock=clock,
    )
    assert result is not None
    assert [method for method, _url in transport.calls] == ["GET", "GET"]
    assert all("/generation?" in url for _method, url in transport.calls)
    assert clock.sleeps == [0.5]


def _init_experiment_repo(
    tmp_path: Path, *, item_count: int = 1, reconciliation_semantics: str | None = None
) -> tuple[Path, Path, Path]:
    root = _GIT_TMP / tmp_path.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    template = root / ".empty-git-template"
    template.mkdir()
    _git(root, "init", "--template", str(template))
    workload = root / "workload.jsonl"
    lines = []
    for index in range(1, item_count + 1):
        item_id = f"or-billing-{index:04d}"
        lines.append(
            json.dumps({"id": item_id, "prompt": f"unique prompt {item_id}", "tags": {"t": "v"}})
        )
    workload.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pricing = root / "pricing-source.json"
    pricing.write_text(
        json.dumps(
            {
                "provider": "openrouter",
                "model": REQUESTED_MODEL,
                "pricing_table_version": "openrouter-gpt-4o-mini-2024-07-18-2026-09-11",
                "pricing_observed_at": "2026-09-11T22:02:53.267052+00:00",
                "source_url": "https://openrouter.ai/api/v1/models/openai/gpt-4o-mini-2024-07-18/endpoints",
                "rates_usd": {
                    "input_per_token": "0.00000015",
                    "cached_input_per_token": "0.000000075",
                    "output_per_token": "0.0000006",
                    "cache_write_per_token": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tariff = load_frozen_tariff(json.loads(pricing.read_text(encoding="utf-8")))
    per = conservative_request_cost_usd(prompt_bytes=3000, max_output_tokens=8, tariff=tariff)
    spec = {
        "experiment_id": EXPERIMENT_ID,
        "classification": "GATEWAY_BILLING_VALIDATION",
        "status": "PRE_REGISTERED_BEFORE_EXECUTION",
        "locked_at_utc": "2026-09-11T22:04:49+00:00",
        "workload": {
            "path": str(workload),
            "sha256": sha256(workload.read_bytes()).hexdigest(),
        },
        "pricing": {"path": str(pricing)},
        "runtime": {
            "execution_gateway": "openrouter",
            "requested_model": REQUESTED_MODEL,
            "application_retries": 0,
            "sdk_retries": 0,
            "allow_fallbacks": False,
            "provider_only": ["openai"],
            "provider_order": ["openai"],
            "stream": False,
            "response_cache_opt_in": False,
            "service_tier": None,
            "max_tokens": 8,
            "temperature": 0,
        },
        "spend_policy": {
            "user_authorized_maximum_usd": "0.20",
            "provider_side_maximum_usd": "0.05",
            "software_guard_usd": "0.025",
            "planned_request_count": item_count,
            "prompt_byte_bound": 3000,
            "max_output_tokens": 8,
            "conservative_per_request_usd": format(per, "f"),
            "conservative_full_run_usd": format(per * item_count, "f"),
        },
        "poll_policy": {
            "generation_metadata_poll": {
                "max_attempts": 3,
                "initial_delay_seconds": 0.01,
                "backoff_multiplier": 2.0,
                "max_delay_seconds": 0.02,
            },
            "key_usage_settlement_poll": {
                "max_attempts": 3,
                "delay_seconds": 0.01,
                "stable_reads": 2,
            },
        },
    }
    if reconciliation_semantics is not None:
        spec["reconciliation_semantics"] = reconciliation_semantics
    spec_path = root / "experiment.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    _git(root, "add", "experiment.json", "pricing-source.json", "workload.jsonl")
    _git(
        root,
        "commit",
        "-m",
        "lock openrouter billing spec",
        env={
            "GIT_AUTHOR_DATE": "2026-09-11T22:04:49+00:00",
            "GIT_COMMITTER_DATE": "2026-09-11T22:04:49+00:00",
        },
    )
    return root, spec_path, root / "out"


def test_runner_blocks_when_key_limit_exceeds_provider_side_maximum(tmp_path: Path) -> None:
    repo, spec_path, output_dir = _init_experiment_repo(tmp_path)
    raw_dir = tmp_path / "raw-outside"
    transport = ScriptedTransport(
        responses=[HttpResponse(status_code=200, body=_key_body(usage="0", limit="0.20", remaining="0.20"))]
    )
    result = run_openrouter_billing_experiment(
        spec_path=spec_path,
        output_dir=output_dir,
        raw_dir=raw_dir,
        repo_root=repo,
        transport=transport,
        env={"OPENROUTER_API_KEY": "test-key"},
        clock=FakeClock(),
        execute_paid=True,
    )
    assert result["classification"] == "BLOCKED"
    assert result["provider_key_limit_usd"] == "0.20"
    assert all(method == "GET" for method, _url in transport.calls)
    assert not any(url.endswith("/chat/completions") for _method, url in transport.calls)


def test_runner_records_one_paid_attempt_without_retry(tmp_path: Path) -> None:
    repo, spec_path, output_dir = _init_experiment_repo(tmp_path)
    raw_dir = tmp_path / "raw-outside"
    cost = "0.0001212"
    transport = ScriptedTransport(
        responses=[
            HttpResponse(status_code=200, body=_key_body(usage="0")),
            HttpResponse(status_code=200, body=_chat_body(cost=cost)),
            HttpResponse(status_code=200, body=_generation_body(total_cost=cost)),
            HttpResponse(status_code=200, body=_key_body(usage=cost, remaining="0.0498788")),
        ]
    )
    result = run_openrouter_billing_experiment(
        spec_path=spec_path,
        output_dir=output_dir,
        raw_dir=raw_dir,
        repo_root=repo,
        transport=transport,
        env={"OPENROUTER_API_KEY": "test-key"},
        clock=FakeClock(),
        execute_paid=True,
    )
    chat_calls = [url for method, url in transport.calls if method == "POST"]
    assert len(chat_calls) == 1
    assert result["retry_count"] == 0
    assert result["successful_request_count"] == 1
    assert result["attempts"][0]["execution_gateway"] == "openrouter"
    assert result["attempts"][0]["requested_model"] == REQUESTED_MODEL
    assert result["attempts"][0]["reported_upstream_provider"] == "OpenAI"
    assert result["direct_openai_billing_validated"] is False
    snapshots = json.loads((output_dir / "key-accounting.json").read_text(encoding="utf-8"))
    dumped = json.dumps(snapshots)
    assert "creator_user_id" not in dumped
    assert "sk-or-" not in dumped
    assert not (output_dir / "pre-key.raw.json").exists()


def test_unexpected_provider_stops_before_second_request(tmp_path: Path) -> None:
    repo, spec_path, output_dir = _init_experiment_repo(tmp_path, item_count=2)
    raw_dir = tmp_path / "raw-outside"
    transport = ScriptedTransport(
        responses=[
            HttpResponse(status_code=200, body=_key_body(usage="0")),
            HttpResponse(status_code=200, body=_chat_body(provider="Azure")),
            HttpResponse(status_code=200, body=_generation_body(provider_name="Azure")),
            HttpResponse(status_code=200, body=_key_body(usage="0.0001212", remaining="0.0498788")),
        ]
    )
    result = run_openrouter_billing_experiment(
        spec_path=spec_path,
        output_dir=output_dir,
        raw_dir=raw_dir,
        repo_root=repo,
        transport=transport,
        env={"OPENROUTER_API_KEY": "test-key"},
        clock=FakeClock(),
        execute_paid=True,
    )
    chat_calls = [url for method, url in transport.calls if method == "POST"]
    assert len(chat_calls) == 1
    assert result["completed_request_count"] == 1
    assert result["classification"] == "INCONCLUSIVE"
    assert result["stop_reason"] == "unexpected_upstream_provider"


def test_failed_paid_request_is_not_retried_or_replaced(tmp_path: Path) -> None:
    repo, spec_path, output_dir = _init_experiment_repo(tmp_path, item_count=2)
    raw_dir = tmp_path / "raw-outside"
    transport = ScriptedTransport(
        responses=[
            HttpResponse(status_code=200, body=_key_body(usage="0")),
            HttpResponse(status_code=500, body=b'{"error":{"message":"upstream"}}'),
            HttpResponse(status_code=200, body=_key_body(usage="0")),
            HttpResponse(status_code=200, body=_key_body(usage="0")),
        ]
    )
    result = run_openrouter_billing_experiment(
        spec_path=spec_path,
        output_dir=output_dir,
        raw_dir=raw_dir,
        repo_root=repo,
        transport=transport,
        env={"OPENROUTER_API_KEY": "test-key"},
        clock=FakeClock(),
        execute_paid=True,
    )
    assert [method for method, url in transport.calls if method == "POST"] == ["POST"]
    assert result["retry_count"] == 0
    assert result["completed_request_count"] == 1
    assert result["successful_request_count"] == 0
    assert result["classification"] == "INCONCLUSIVE"


def test_runner_refuses_raw_dir_inside_git(tmp_path: Path) -> None:
    repo, spec_path, output_dir = _init_experiment_repo(tmp_path)
    transport = ScriptedTransport(responses=[])
    with pytest.raises(OpenRouterBillingError, match="outside the Git repository"):
        run_openrouter_billing_experiment(
            spec_path=spec_path,
            output_dir=output_dir,
            raw_dir=repo / "raw",
            repo_root=repo,
            transport=transport,
            env={"OPENROUTER_API_KEY": "test-key"},
            clock=FakeClock(),
            execute_paid=True,
        )


def test_historical_mission_artifacts_remain_untouched() -> None:
    mission4 = REPO_ROOT / "benchmarks/reports/local-json-mode-20260911/decision.json"
    mission5 = REPO_ROOT / "benchmarks/reports/ifstruct-json-mode-20260911/decision.json"
    mission6a = REPO_ROOT / "docs/10_PROVIDER_BILLING_VALIDATION_PROTOCOL.md"
    mission6b = REPO_ROOT / "benchmarks/reports/openrouter-billing-reconciliation-6b-20260911T220449Z/decision.json"
    assert mission4.is_file()
    assert mission5.is_file()
    assert mission6a.is_file()
    assert "GATEWAY_BILLING_VALIDATION" not in mission4.read_text(encoding="utf-8")
    assert "VALIDATED_MATCH" not in mission5.read_text(encoding="utf-8")
    decision = json.loads(mission6b.read_text(encoding="utf-8"))
    assert decision["classification"] == "VALIDATED_MISMATCH"


def _matching_cost(prompt_tokens: int = 80) -> str:
    tariff = load_frozen_tariff(
        {
            "provider": "openrouter",
            "model": REQUESTED_MODEL,
            "pricing_table_version": "openrouter-gpt-4o-mini-2024-07-18-2026-09-11",
            "pricing_observed_at": "2026-09-11T22:02:53.267052+00:00",
            "source_url": "https://openrouter.ai/api/v1/models/openai/gpt-4o-mini-2024-07-18/endpoints",
            "rates_usd": {
                "input_per_token": "0.00000015",
                "cached_input_per_token": "0.000000075",
                "output_per_token": "0.0000006",
                "cache_write_per_token": None,
            },
        }
    )
    cost = reconstruct_openrouter_cost(
        prompt_tokens=prompt_tokens,
        completion_tokens=2,
        cached_input_tokens=0,
        cache_write_tokens=0,
        tariff=tariff,
    )
    assert cost is not None
    return format(cost, "f")


def test_v1_semantics_still_mismatch_when_tokens_prompt_differs(tmp_path: Path) -> None:
    repo, spec_path, output_dir = _init_experiment_repo(tmp_path)
    raw_dir = tmp_path / "raw-outside"
    cost = _matching_cost(80)
    transport = ScriptedTransport(
        responses=[
            HttpResponse(status_code=200, body=_key_body(usage="0")),
            HttpResponse(
                status_code=200,
                body=_chat_body(cost=cost, prompt_tokens=80, response_upstream_cost=cost),
            ),
            HttpResponse(
                status_code=200,
                body=_generation_body(
                    total_cost=cost,
                    prompt_tokens=80,
                    tokens_prompt=64,
                    native_tokens_prompt=80,
                    upstream_inference_cost="0",
                ),
            ),
            HttpResponse(status_code=200, body=_key_body(usage=cost, remaining="0.0498788")),
        ]
    )
    result = run_openrouter_billing_experiment(
        spec_path=spec_path,
        output_dir=output_dir,
        raw_dir=raw_dir,
        repo_root=repo,
        transport=transport,
        env={"OPENROUTER_API_KEY": "test-key"},
        clock=FakeClock(),
        execute_paid=True,
    )
    rules = {item["rule_id"]: item["status"] for item in result["rules"]}
    assert result["reconciliation_semantics"] == "v1"
    assert rules["R1"] == "MISMATCH"
    assert result["classification"] == "VALIDATED_MISMATCH"
    assert result["attempts"][0]["response_upstream_inference_cost"] == cost
    assert result["attempts"][0]["generation_upstream_inference_cost"] == "0"
    assert result["streamed_field_interpretation"]["proves_client_visible_streaming"] is False
    assert result["streamed_field_interpretation"]["classification"] == "PROVIDER_METADATA_CONTRACT_ANOMALY"


def test_v2_semantics_does_not_mismatch_unknown_tokens_prompt(tmp_path: Path) -> None:
    repo, spec_path, output_dir = _init_experiment_repo(tmp_path, reconciliation_semantics="v2")
    raw_dir = tmp_path / "raw-outside"
    cost = _matching_cost(80)
    transport = ScriptedTransport(
        responses=[
            HttpResponse(status_code=200, body=_key_body(usage="0")),
            HttpResponse(
                status_code=200,
                body=_chat_body(cost=cost, prompt_tokens=80, response_upstream_cost=cost),
            ),
            HttpResponse(
                status_code=200,
                body=_generation_body(
                    total_cost=cost,
                    prompt_tokens=80,
                    tokens_prompt=64,
                    native_tokens_prompt=80,
                    upstream_inference_cost="0",
                ),
            ),
            HttpResponse(status_code=200, body=_key_body(usage=cost, remaining="0.0498788")),
        ]
    )
    result = run_openrouter_billing_experiment(
        spec_path=spec_path,
        output_dir=output_dir,
        raw_dir=raw_dir,
        repo_root=repo,
        transport=transport,
        env={"OPENROUTER_API_KEY": "test-key"},
        clock=FakeClock(),
        execute_paid=True,
    )
    rules = {item["rule_id"]: item["status"] for item in result["rules"]}
    assert result["reconciliation_semantics"] == "v2"
    assert rules["R1"] == "MATCH"
    assert rules["R2"] == "MATCH"
    assert result["classification"] == "VALIDATED_MATCH"
    assert result["response_upstream_inference_cost_sum"] == cost
    assert result["generation_upstream_inference_cost_sum"] == "0"
