"""OpenRouter gateway billing reconciliation for Mission 6B-OR.

This module is experiment evidence, not a general provider-adapter expansion. It prices
OpenRouter executions with a frozen OpenRouter tariff and never inherits direct-OpenAI
identity or the production ``openai-standard-*`` pricing table.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .experiment_provenance import ExperimentProvenanceError

EXECUTION_GATEWAY = "openrouter"
REQUESTED_MODEL = "openai/gpt-4o-mini-2024-07-18"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_COMPLETIONS_URL = f"{OPENROUTER_API_BASE}/chat/completions"
OPENROUTER_KEY_URL = f"{OPENROUTER_API_BASE}/key"
OPENROUTER_GENERATION_URL = f"{OPENROUTER_API_BASE}/generation"
ALLOWED_UPSTREAM_PROVIDER_NAMES = frozenset({"openai", "OpenAI"})
ALLOWED_SERVICE_TIERS = frozenset({"", "default", "standard"})
WORKLOAD_IDENTITY = "openrouter-billing-economics-n50"
WORKLOAD_ITEM_COUNT = 50
PROMPT_BYTE_BOUND = 3000
MAX_OUTPUT_TOKENS = 8
EXPERIMENT_ID_PREFIX = "openrouter-billing-reconciliation-6b-"
EXPERIMENT_ID_PATTERN = re.compile(r"^openrouter-billing-reconciliation-6b-\d{8}T\d{6}Z$")
USER_AUTHORIZED_MAXIMUM_USD = Decimal("0.20")
PROVIDER_SIDE_MAXIMUM_USD = Decimal("0.05")
SOFTWARE_GUARD_USD = Decimal("0.025")
CLASSIFICATION = "GATEWAY_BILLING_VALIDATION"

_SECRET_MARKERS = ("sk-or-", "sk-ant-", "Bearer ")
_KEY_SANITIZE_KEEP = frozenset(
    {
        "usage",
        "usage_daily",
        "usage_weekly",
        "usage_monthly",
        "byok_usage",
        "byok_usage_daily",
        "byok_usage_weekly",
        "byok_usage_monthly",
        "limit",
        "limit_remaining",
        "limit_reset",
        "include_byok_in_limit",
        "is_free_tier",
        "is_management_key",
        "is_provisioning_key",
        "expires_at",
    }
)
_GENERATION_SANITIZE_KEEP = frozenset(
    {
        "id",
        "created_at",
        "model",
        "provider_name",
        "request_id",
        "upstream_id",
        "tokens_prompt",
        "tokens_completion",
        "native_tokens_prompt",
        "native_tokens_cached",
        "native_tokens_completion",
        "native_tokens_reasoning",
        "latency",
        "generation_time",
        "service_tier",
        "total_cost",
        "upstream_inference_cost",
        "cache_discount",
        "streamed",
        "cancelled",
        "finish_reason",
        "native_finish_reason",
        "is_byok",
        "usage",
        "api_type",
    }
)
_CHAT_USAGE_KEEP = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "is_byok",
        "prompt_tokens_details",
        "completion_tokens_details",
        "cost_details",
    }
)


class EvidenceClass(StrEnum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class ExperimentClassification(StrEnum):
    VALIDATED_MATCH = "VALIDATED_MATCH"
    VALIDATED_MISMATCH = "VALIDATED_MISMATCH"
    INCONCLUSIVE = "INCONCLUSIVE"
    BLOCKED = "BLOCKED"


class KeyReadinessStatus(StrEnum):
    READY = "READY"
    READY_AFTER_USER_ACTION = "READY_AFTER_USER_ACTION"
    BLOCKED = "BLOCKED"


class OpenRouterBillingError(ValueError):
    """Fail-closed error for OpenRouter billing-reconciliation logic."""


class OpenRouterTransportError(RuntimeError):
    """Network/transport failure without embedding secrets."""


@dataclass(frozen=True)
class FrozenTariff:
    """Immutable OpenRouter tariff used to reconstruct cost B."""

    provider: str
    model: str
    pricing_table_version: str
    pricing_observed_at: str
    source_url: str
    input_per_token: Decimal
    cached_input_per_token: Decimal
    output_per_token: Decimal
    cache_write_per_token: Decimal | None

    def __post_init__(self) -> None:
        if self.provider != EXECUTION_GATEWAY:
            raise OpenRouterBillingError("frozen tariff provider must be openrouter")
        if self.model != REQUESTED_MODEL:
            raise OpenRouterBillingError("frozen tariff model must be the pinned snapshot")
        rates = (self.input_per_token, self.cached_input_per_token, self.output_per_token)
        if min(rates) < 0:
            raise OpenRouterBillingError("tariff rates must be non-negative")
        if self.cache_write_per_token is not None and self.cache_write_per_token < 0:
            raise OpenRouterBillingError("cache-write tariff must be non-negative when listed")


@dataclass(frozen=True)
class SpendPolicy:
    """Three independent spend ceilings plus frozen conservative bounds."""

    user_authorized_maximum_usd: Decimal
    provider_side_maximum_usd: Decimal
    software_guard_usd: Decimal
    planned_request_count: int
    prompt_byte_bound: int
    max_output_tokens: int
    conservative_per_request_usd: Decimal
    conservative_full_run_usd: Decimal

    def __post_init__(self) -> None:
        if self.user_authorized_maximum_usd != USER_AUTHORIZED_MAXIMUM_USD:
            raise OpenRouterBillingError("user-authorized maximum must be USD 0.20")
        if self.provider_side_maximum_usd != PROVIDER_SIDE_MAXIMUM_USD:
            raise OpenRouterBillingError("provider-side key maximum must be USD 0.05")
        if self.software_guard_usd != SOFTWARE_GUARD_USD:
            raise OpenRouterBillingError("software spend guard must be USD 0.025")
        if self.provider_side_maximum_usd > self.user_authorized_maximum_usd:
            raise OpenRouterBillingError("provider-side maximum cannot exceed user authorization")
        if self.software_guard_usd >= self.provider_side_maximum_usd:
            raise OpenRouterBillingError("software guard must be below the provider-side key maximum")
        if self.planned_request_count < 1:
            raise OpenRouterBillingError("planned request count must be at least 1")
        if self.prompt_byte_bound < 1:
            raise OpenRouterBillingError("prompt byte bound must be at least 1")
        if self.max_output_tokens < 1:
            raise OpenRouterBillingError("max output tokens must be at least 1")
        if self.conservative_per_request_usd <= 0 or self.conservative_full_run_usd <= 0:
            raise OpenRouterBillingError("conservative spend bounds must be positive")
        expected_full = self.conservative_per_request_usd * self.planned_request_count
        if expected_full != self.conservative_full_run_usd:
            raise OpenRouterBillingError("full-run conservative bound must equal per-request bound × count")
        if self.conservative_full_run_usd >= self.software_guard_usd:
            raise OpenRouterBillingError("conservative full-run maximum must be below USD 0.025")


@dataclass(frozen=True)
class KeyReadiness:
    status: KeyReadinessStatus
    ready: bool
    user_action: str | None
    limit: Decimal | None
    limit_remaining: Decimal | None
    usage: Decimal | None
    limit_reset: str | None
    is_free_tier: bool | None
    is_management_key: bool | None
    is_provisioning_key: bool | None
    byok_usage: Decimal | None


@dataclass(frozen=True)
class ResponseUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_input_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None
    cost: Decimal | None
    upstream_inference_cost: Decimal | None
    is_byok: bool | None


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    required: bool
    status: EvidenceClass
    reason: str
    left: str | None = None
    right: str | None = None


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: bytes


class HttpTransport(Protocol):
    def send(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> HttpResponse:
        """Send one HTTP request. Implementations must not log Authorization headers."""


class Clock(Protocol):
    def now_utc(self) -> datetime: ...

    def sleep(self, seconds: float) -> None: ...


@dataclass(frozen=True)
class SystemClock:
    def now_utc(self) -> datetime:
        return datetime.now(tz=UTC)

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise OpenRouterBillingError("sleep duration cannot be negative")
        import time

        time.sleep(seconds)


def make_experiment_id(now: datetime) -> str:
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{EXPERIMENT_ID_PREFIX}{stamp}"


def require_experiment_id(experiment_id: str) -> str:
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise OpenRouterBillingError("experiment_id must be openrouter-billing-reconciliation-6b-YYYYMMDDTHHMMSSZ")
    return experiment_id


def require_openrouter_gateway(execution_gateway: str) -> None:
    if execution_gateway != EXECUTION_GATEWAY:
        raise OpenRouterBillingError(
            "Mission 6B-OR execution gateway must be openrouter, not a direct OpenAI identity"
        )


def require_openrouter_model(model: str) -> None:
    if model != REQUESTED_MODEL:
        raise OpenRouterBillingError("requested model must be the frozen OpenRouter snapshot")


def env_key_presence(env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    value = source.get("OPENROUTER_API_KEY")
    if value is None or not value.strip():
        return "ABSENT"
    return "PRESENT"


def require_present_openrouter_key(env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    if env_key_presence(source) != "PRESENT":
        raise OpenRouterBillingError("OPENROUTER_API_KEY is ABSENT")
    return source["OPENROUTER_API_KEY"]


def load_json_preserving_decimals(raw: str | bytes) -> object:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return json.loads(text, parse_float=Decimal)


def decimal_from_json_number(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise OpenRouterBillingError(f"{field} is not a JSON number")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        return Decimal(value)
    raise OpenRouterBillingError(f"{field} must be parsed from JSON decimal text, not a binary float")


def optional_decimal(value: object, *, field: str) -> Decimal | None:
    if value is None:
        return None
    return decimal_from_json_number(value, field=field)


def format_decimal(value: Decimal) -> str:
    return format(value, "f")


def load_frozen_tariff(payload: Mapping[str, Any]) -> FrozenTariff:
    require_openrouter_gateway(str(payload["provider"]))
    require_openrouter_model(str(payload["model"]))
    rates = payload["rates_usd"]
    if not isinstance(rates, Mapping):
        raise OpenRouterBillingError("pricing rates_usd must be an object")
    cache_write = rates.get("cache_write_per_token")
    return FrozenTariff(
        provider=str(payload["provider"]),
        model=str(payload["model"]),
        pricing_table_version=str(payload["pricing_table_version"]),
        pricing_observed_at=str(payload["pricing_observed_at"]),
        source_url=str(payload["source_url"]),
        input_per_token=Decimal(str(rates["input_per_token"])),
        cached_input_per_token=Decimal(str(rates["cached_input_per_token"])),
        output_per_token=Decimal(str(rates["output_per_token"])),
        cache_write_per_token=None if cache_write is None else Decimal(str(cache_write)),
    )


def load_spend_policy(payload: Mapping[str, Any]) -> SpendPolicy:
    return SpendPolicy(
        user_authorized_maximum_usd=Decimal(str(payload["user_authorized_maximum_usd"])),
        provider_side_maximum_usd=Decimal(str(payload["provider_side_maximum_usd"])),
        software_guard_usd=Decimal(str(payload["software_guard_usd"])),
        planned_request_count=int(payload["planned_request_count"]),
        prompt_byte_bound=int(payload["prompt_byte_bound"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        conservative_per_request_usd=Decimal(str(payload["conservative_per_request_usd"])),
        conservative_full_run_usd=Decimal(str(payload["conservative_full_run_usd"])),
    )


def conservative_request_cost_usd(
    *,
    prompt_bytes: int,
    max_output_tokens: int,
    tariff: FrozenTariff,
) -> Decimal:
    if prompt_bytes < 0 or max_output_tokens < 0:
        raise OpenRouterBillingError("conservative request bounds cannot be negative")
    return Decimal(prompt_bytes) * tariff.input_per_token + Decimal(max_output_tokens) * tariff.output_per_token


def conservative_full_run_cost_usd(
    *,
    prompt_byte_bound: int,
    request_count: int,
    max_output_tokens: int,
    tariff: FrozenTariff,
) -> Decimal:
    if request_count < 1:
        raise OpenRouterBillingError("request count must be at least 1")
    per_request = conservative_request_cost_usd(
        prompt_bytes=prompt_byte_bound,
        max_output_tokens=max_output_tokens,
        tariff=tariff,
    )
    return per_request * request_count


def require_conservative_budget(tariff: FrozenTariff, spend: SpendPolicy) -> None:
    computed_per = conservative_request_cost_usd(
        prompt_bytes=spend.prompt_byte_bound,
        max_output_tokens=spend.max_output_tokens,
        tariff=tariff,
    )
    computed_full = computed_per * spend.planned_request_count
    if computed_per != spend.conservative_per_request_usd:
        raise OpenRouterBillingError("frozen per-request conservative bound does not match tariff arithmetic")
    if computed_full != spend.conservative_full_run_usd:
        raise OpenRouterBillingError("frozen full-run conservative bound does not match tariff arithmetic")
    if computed_full >= spend.software_guard_usd:
        raise OpenRouterBillingError("conservative full-run maximum is not below the software guard")


def reconstruct_openrouter_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_input_tokens: int | None,
    cache_write_tokens: int | None,
    tariff: FrozenTariff,
) -> Decimal | None:
    if min(prompt_tokens, completion_tokens) < 0:
        raise OpenRouterBillingError("token counts must be non-negative")
    if cached_input_tokens is None:
        return None
    if cached_input_tokens < 0:
        raise OpenRouterBillingError("cached input tokens must be non-negative")
    if cached_input_tokens > prompt_tokens:
        raise OpenRouterBillingError("cached input tokens cannot exceed prompt tokens")
    write_cost = Decimal("0")
    if cache_write_tokens is not None:
        if cache_write_tokens < 0:
            raise OpenRouterBillingError("cache-write tokens must be non-negative")
        if cache_write_tokens > 0 and tariff.cache_write_per_token is None:
            return None
        if tariff.cache_write_per_token is not None:
            write_cost = Decimal(cache_write_tokens) * tariff.cache_write_per_token
    billable_input = prompt_tokens - cached_input_tokens
    return (
        Decimal(billable_input) * tariff.input_per_token
        + Decimal(cached_input_tokens) * tariff.cached_input_per_token
        + Decimal(completion_tokens) * tariff.output_per_token
        + write_cost
    )


def budget_allows_next_request(
    cumulative: Decimal,
    next_request_conservative: Decimal,
    policy: SpendPolicy,
) -> tuple[bool, str]:
    if cumulative < 0 or next_request_conservative < 0:
        raise OpenRouterBillingError("budget amounts cannot be negative")
    if cumulative >= policy.software_guard_usd:
        return False, "software spend guard reached"
    if cumulative >= policy.user_authorized_maximum_usd:
        return False, "user-authorized maximum reached"
    projected = cumulative + next_request_conservative
    if projected >= policy.software_guard_usd:
        return False, "next request could reach the software spend guard"
    if projected >= policy.user_authorized_maximum_usd:
        return False, "next request could reach the user-authorized maximum"
    return True, "ok"


def observed_cost_for_budget(response_cost: Decimal | None, fallback: Decimal) -> Decimal:
    if response_cost is None:
        return fallback
    if response_cost < 0:
        raise OpenRouterBillingError("response cost cannot be negative")
    return response_cost


def assess_key_readiness(payload: Mapping[str, Any], *, spend: SpendPolicy) -> KeyReadiness:
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return _blocked_key("Key metadata was not a JSON object; confirm the dedicated OpenRouter key.")
    if data.get("is_provisioning_key") is True:
        return _blocked_key(
            "This key is a provisioning key. Use the dedicated inference key with a USD 0.05 limit."
        )
    if data.get("is_management_key") is True:
        return _blocked_key(
            "This key is a management key. Use the dedicated inference key with a USD 0.05 limit."
        )
    try:
        limit = optional_decimal(data.get("limit"), field="limit")
        remaining = optional_decimal(data.get("limit_remaining"), field="limit_remaining")
        usage = optional_decimal(data.get("usage"), field="usage")
        byok_usage = optional_decimal(data.get("byok_usage"), field="byok_usage") if "byok_usage" in data else None
    except OpenRouterBillingError:
        return _blocked_key("Key metadata numeric fields could not be parsed as decimals.")
    limit_reset = data.get("limit_reset")
    reset_text = None if limit_reset is None else str(limit_reset)
    free_tier = data.get("is_free_tier")
    is_free_tier = free_tier if isinstance(free_tier, bool) else None
    management = data.get("is_management_key")
    provisioning = data.get("is_provisioning_key")
    is_management_key = management if isinstance(management, bool) else None
    is_provisioning_key = provisioning if isinstance(provisioning, bool) else None

    def blocked(action: str) -> KeyReadiness:
        return _blocked_key(
            action,
            limit=limit,
            limit_remaining=remaining,
            usage=usage,
            limit_reset=reset_text,
            is_free_tier=is_free_tier,
            is_management_key=is_management_key,
            is_provisioning_key=is_provisioning_key,
            byok_usage=byok_usage,
        )

    if limit is None:
        return blocked(
            "Set a per-key spending limit of at most USD 0.05 on this dedicated OpenRouter key. "
            "Do not modify the key from this experiment."
        )
    if limit > spend.provider_side_maximum_usd:
        return blocked(
            "Lower this dedicated OpenRouter key's spending limit to at most USD 0.05. "
            "Do not modify the key from this experiment."
        )
    if remaining is None:
        return blocked("limit_remaining could not be read; refusing inference.")
    if remaining <= spend.conservative_full_run_usd:
        return blocked(
            "Insufficient limit_remaining for the frozen conservative planned spend. "
            "Do not raise the key limit above USD 0.05."
        )
    if usage is None:
        return blocked("Key cumulative usage could not be read; refusing inference.")
    if usage != 0:
        return blocked(
            "Dedicated-key usage is not zero before the experiment. Refusing a contaminated baseline."
        )
    if byok_usage is not None and byok_usage != 0:
        return blocked(
            "Key metadata reports non-zero BYOK usage. This mission must not mix BYOK traffic."
        )
    return KeyReadiness(
        status=KeyReadinessStatus.READY,
        ready=True,
        user_action=None,
        limit=limit,
        limit_remaining=remaining,
        usage=usage,
        limit_reset=reset_text,
        is_free_tier=is_free_tier,
        is_management_key=is_management_key,
        is_provisioning_key=is_provisioning_key,
        byok_usage=byok_usage,
    )


def _blocked_key(
    action: str,
    *,
    limit: Decimal | None = None,
    limit_remaining: Decimal | None = None,
    usage: Decimal | None = None,
    limit_reset: str | None = None,
    is_free_tier: bool | None = None,
    is_management_key: bool | None = None,
    is_provisioning_key: bool | None = None,
    byok_usage: Decimal | None = None,
) -> KeyReadiness:
    return KeyReadiness(
        status=KeyReadinessStatus.READY_AFTER_USER_ACTION,
        ready=False,
        user_action=action,
        limit=limit,
        limit_remaining=limit_remaining,
        usage=usage,
        limit_reset=limit_reset,
        is_free_tier=is_free_tier,
        is_management_key=is_management_key,
        is_provisioning_key=is_provisioning_key,
        byok_usage=byok_usage,
    )


def sanitize_key_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return {"data": {}}
    kept = {key: _copy_jsonable(data[key]) for key in _KEY_SANITIZE_KEEP if key in data}
    return {"data": _redact_secret_strings(kept)}


def sanitize_generation_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return {"data": {}}
    kept = {key: _copy_jsonable(data[key]) for key in _GENERATION_SANITIZE_KEEP if key in data}
    return {"data": _redact_secret_strings(kept)}


def sanitize_chat_completion_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key in ("id", "object", "created", "model", "provider", "system_fingerprint"):
        if key in payload:
            sanitized[key] = _copy_jsonable(payload[key])
    usage = payload.get("usage")
    if isinstance(usage, Mapping):
        sanitized["usage"] = {key: _copy_jsonable(usage[key]) for key in _CHAT_USAGE_KEEP if key in usage}
    choices = payload.get("choices")
    if isinstance(choices, list):
        sanitized["choices"] = [
            {
                "index": item.get("index"),
                "finish_reason": item.get("finish_reason"),
                "native_finish_reason": item.get("native_finish_reason"),
            }
            for item in choices
            if isinstance(item, Mapping)
        ]
    redacted = _redact_secret_strings(sanitized)
    if not isinstance(redacted, dict):
        raise OpenRouterBillingError("sanitized chat payload must be an object")
    return cast(dict[str, Any], redacted)


def extract_response_usage(payload: Mapping[str, Any]) -> ResponseUsage | None:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None
    try:
        prompt_tokens = _require_int(usage.get("prompt_tokens"), field="prompt_tokens")
        completion_tokens = _require_int(usage.get("completion_tokens"), field="completion_tokens")
        total_tokens = _require_int(usage.get("total_tokens"), field="total_tokens")
    except OpenRouterBillingError:
        return None
    details = usage.get("prompt_tokens_details")
    cached = None
    cache_write = None
    if isinstance(details, Mapping):
        if "cached_tokens" in details and details.get("cached_tokens") is not None:
            cached = _require_int(details.get("cached_tokens"), field="cached_tokens")
        if "cache_write_tokens" in details and details.get("cache_write_tokens") is not None:
            cache_write = _require_int(details.get("cache_write_tokens"), field="cache_write_tokens")
    completion_details = usage.get("completion_tokens_details")
    reasoning = None
    if isinstance(completion_details, Mapping) and completion_details.get("reasoning_tokens") is not None:
        reasoning = _require_int(completion_details.get("reasoning_tokens"), field="reasoning_tokens")
    cost = optional_decimal(usage.get("cost"), field="usage.cost") if "cost" in usage else None
    cost_details = usage.get("cost_details")
    upstream = None
    if isinstance(cost_details, Mapping) and "upstream_inference_cost" in cost_details:
        upstream = optional_decimal(
            cost_details.get("upstream_inference_cost"),
            field="usage.cost_details.upstream_inference_cost",
        )
    is_byok = usage.get("is_byok")
    return ResponseUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        cost=cost,
        upstream_inference_cost=upstream,
        is_byok=is_byok if isinstance(is_byok, bool) else None,
    )


def allowed_upstream_provider(name: str | None) -> bool:
    if name is None:
        return False
    return name.strip() in ALLOWED_UPSTREAM_PROVIDER_NAMES or name.strip().lower() == "openai"


def allowed_service_tier(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in ALLOWED_SERVICE_TIERS


def chat_completion_headers(api_key: str) -> dict[str, str]:
    if not api_key.strip():
        raise OpenRouterBillingError("OPENROUTER_API_KEY is empty")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def chat_completion_body(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    provider_only: Sequence[str],
    provider_order: Sequence[str],
    allow_fallbacks: bool,
) -> dict[str, Any]:
    require_openrouter_model(model)
    if allow_fallbacks:
        raise OpenRouterBillingError("OpenRouter provider fallbacks must be disabled")
    if list(provider_only) != ["openai"] or list(provider_order) != ["openai"]:
        raise OpenRouterBillingError("provider only/order must be exactly ['openai']")
    if max_tokens < 1:
        raise OpenRouterBillingError("max_tokens must be at least 1")
    if temperature != 0:
        raise OpenRouterBillingError("temperature must be 0")
    utf8_len = len(prompt.encode("utf-8"))
    if utf8_len > PROMPT_BYTE_BOUND:
        raise OpenRouterBillingError("serialized user prompt exceeds the 3000-byte bound")
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "provider": {"only": ["openai"], "order": ["openai"], "allow_fallbacks": False},
    }


def compare_optional_int(left: int | None, right: int | None) -> EvidenceClass:
    if left is None or right is None:
        return EvidenceClass.INCONCLUSIVE
    if left == right:
        return EvidenceClass.MATCH
    return EvidenceClass.MISMATCH


def combine_required_statuses(statuses: Sequence[EvidenceClass]) -> EvidenceClass:
    if not statuses:
        return EvidenceClass.INCONCLUSIVE
    if any(status == EvidenceClass.MISMATCH for status in statuses):
        return EvidenceClass.MISMATCH
    if any(status == EvidenceClass.INCONCLUSIVE for status in statuses):
        return EvidenceClass.INCONCLUSIVE
    if any(status == EvidenceClass.NOT_COMPARABLE for status in statuses):
        return EvidenceClass.NOT_COMPARABLE
    if all(status == EvidenceClass.MATCH for status in statuses):
        return EvidenceClass.MATCH
    return EvidenceClass.INCONCLUSIVE


def reconcile_tokens_r1(
    *,
    response: ResponseUsage | None,
    generation: Mapping[str, Any] | None,
) -> RuleResult:
    """Historical Mission 6B-OR R1 (v1). Do not change; v2 lives in evidence semantics."""
    if response is None or generation is None:
        return RuleResult(
            rule_id="R1",
            required=True,
            status=EvidenceClass.INCONCLUSIVE,
            reason="response usage or generation metadata is missing",
        )
    comparable = [
        compare_optional_int(response.prompt_tokens, _optional_int(generation.get("tokens_prompt"))),
        compare_optional_int(
            response.completion_tokens, _optional_int(generation.get("tokens_completion"))
        ),
    ]
    native_cached = _optional_int(generation.get("native_tokens_cached"))
    if response.cached_input_tokens is not None and native_cached is not None:
        comparable.append(compare_optional_int(response.cached_input_tokens, native_cached))
    status = combine_required_statuses(comparable)
    native_prompt = _optional_int(generation.get("native_tokens_prompt"))
    billed_prompt = _optional_int(generation.get("tokens_prompt"))
    note = "comparable billed token fields compared with exact integer equality"
    if native_prompt is not None and billed_prompt is not None and native_prompt != billed_prompt:
        note += "; native_tokens_prompt differs from tokens_prompt and is preserved as NOT_COMPARABLE"
    return RuleResult(
        rule_id="R1",
        required=True,
        status=status,
        reason=note,
        left=str(response.prompt_tokens),
        right=str(generation.get("tokens_prompt")),
    )


def reconcile_reconstructed_vs_response_r2(
    *,
    reconstructed: Decimal | None,
    response_cost: Decimal | None,
    cache_discount: Decimal | None,
) -> RuleResult:
    if reconstructed is None or response_cost is None:
        return RuleResult(
            rule_id="R2",
            required=True,
            status=EvidenceClass.INCONCLUSIVE,
            reason="reconstructed cost or response usage.cost is missing",
        )
    if cache_discount is not None and cache_discount != 0 and reconstructed != response_cost:
        return RuleResult(
            rule_id="R2",
            required=True,
            status=EvidenceClass.NOT_COMPARABLE,
            reason="non-zero cache_discount cannot be reconstructed from the frozen token tariff",
            left=format_decimal(reconstructed),
            right=format_decimal(response_cost),
        )
    if reconstructed == response_cost:
        return RuleResult(
            rule_id="R2",
            required=True,
            status=EvidenceClass.MATCH,
            reason="exact Decimal equality of reconstructed tariff cost and response usage.cost",
            left=format_decimal(reconstructed),
            right=format_decimal(response_cost),
        )
    return RuleResult(
        rule_id="R2",
        required=True,
        status=EvidenceClass.MISMATCH,
        reason="reconstructed tariff cost disagrees with response usage.cost",
        left=format_decimal(reconstructed),
        right=format_decimal(response_cost),
    )


def reconcile_response_vs_generation_cost_r3(
    *,
    response_cost: Decimal | None,
    generation_total_cost: Decimal | None,
) -> RuleResult:
    if response_cost is None or generation_total_cost is None:
        return RuleResult(
            rule_id="R3",
            required=True,
            status=EvidenceClass.INCONCLUSIVE,
            reason="response usage.cost or generation total_cost is missing",
        )
    if response_cost == generation_total_cost:
        return RuleResult(
            rule_id="R3",
            required=True,
            status=EvidenceClass.MATCH,
            reason="exact Decimal equality of response usage.cost and generation total_cost",
            left=format_decimal(response_cost),
            right=format_decimal(generation_total_cost),
        )
    return RuleResult(
        rule_id="R3",
        required=True,
        status=EvidenceClass.MISMATCH,
        reason="response usage.cost disagrees with generation total_cost",
        left=format_decimal(response_cost),
        right=format_decimal(generation_total_cost),
    )


def reconcile_sum_vs_key_delta(
    *,
    rule_id: str,
    amount_sum: Decimal | None,
    key_delta: Decimal | None,
    settled: bool,
    other_traffic_suspected: bool,
) -> RuleResult:
    if amount_sum is None or key_delta is None:
        return RuleResult(
            rule_id=rule_id,
            required=True,
            status=EvidenceClass.INCONCLUSIVE,
            reason="cost sum or key usage delta is missing",
        )
    if other_traffic_suspected:
        return RuleResult(
            rule_id=rule_id,
            required=True,
            status=EvidenceClass.INCONCLUSIVE,
            reason="dedicated-key usage indicates possible other traffic during the window",
            left=format_decimal(amount_sum),
            right=format_decimal(key_delta),
        )
    if amount_sum == key_delta:
        return RuleResult(
            rule_id=rule_id,
            required=True,
            status=EvidenceClass.MATCH,
            reason="exact Decimal equality of summed OpenRouter costs and dedicated-key usage delta",
            left=format_decimal(amount_sum),
            right=format_decimal(key_delta),
        )
    if not settled:
        return RuleResult(
            rule_id=rule_id,
            required=True,
            status=EvidenceClass.INCONCLUSIVE,
            reason="key usage did not settle to an exact match within the polling window",
            left=format_decimal(amount_sum),
            right=format_decimal(key_delta),
        )
    return RuleResult(
        rule_id=rule_id,
        required=True,
        status=EvidenceClass.MISMATCH,
        reason="settled dedicated-key usage delta disagrees with the summed OpenRouter costs",
        left=format_decimal(amount_sum),
        right=format_decimal(key_delta),
    )


def classify_experiment(
    rules: Sequence[RuleResult],
    *,
    blocked: bool,
    completed_requests: int,
    planned_requests: int,
    successful_paid_requests: int,
    protocol_violation: bool = False,
) -> ExperimentClassification:
    if blocked and successful_paid_requests == 0:
        return ExperimentClassification.BLOCKED
    required = [rule for rule in rules if rule.required]
    if any(rule.status == EvidenceClass.MISMATCH for rule in required):
        return ExperimentClassification.VALIDATED_MISMATCH
    if protocol_violation or completed_requests != planned_requests:
        return ExperimentClassification.INCONCLUSIVE
    if successful_paid_requests != planned_requests:
        return ExperimentClassification.INCONCLUSIVE
    if any(
        rule.status in {EvidenceClass.INCONCLUSIVE, EvidenceClass.NOT_COMPARABLE} for rule in required
    ):
        return ExperimentClassification.INCONCLUSIVE
    if required and all(rule.status == EvidenceClass.MATCH for rule in required):
        return ExperimentClassification.VALIDATED_MATCH
    return ExperimentClassification.INCONCLUSIVE


def generate_economics_prompt(item_id: str, experiment_id: str) -> str:
    require_experiment_id(experiment_id)
    nonce = sha256(f"{experiment_id}:{item_id}".encode()).hexdigest()
    return (
        f"InferenceLedger {experiment_id} item {item_id}. "
        "Synthetic economics probe, not a quality evaluation. "
        f"Unique nonce {nonce}. Reply with the single word OK."
    )


def generate_workload_items(experiment_id: str) -> list[dict[str, Any]]:
    require_experiment_id(experiment_id)
    items: list[dict[str, Any]] = []
    for index in range(1, WORKLOAD_ITEM_COUNT + 1):
        item_id = f"or-billing-{index:04d}"
        prompt = generate_economics_prompt(item_id, experiment_id)
        if len(prompt.encode("utf-8")) > PROMPT_BYTE_BOUND:
            raise OpenRouterBillingError("generated prompt exceeds the 3000-byte bound")
        items.append(
            {
                "id": item_id,
                "prompt": prompt,
                "tags": {
                    "experiment": experiment_id,
                    "classification": CLASSIFICATION,
                    "not_quality_eval": "true",
                },
            }
        )
    return items


def render_workload_jsonl(experiment_id: str) -> str:
    return "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        for item in generate_workload_items(experiment_id)
    )


def workload_sha256_hex(experiment_id: str) -> str:
    return sha256(render_workload_jsonl(experiment_id).encode("utf-8")).hexdigest()


def prompt_utf8_len(prompt: str) -> int:
    return len(prompt.encode("utf-8"))


def assert_raw_dir_outside_git(raw_root: Path, repo_root: Path) -> None:
    try:
        raw_root.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return
    raise OpenRouterBillingError("raw evidence directory must be outside the Git repository")


def write_private_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.chmod(path, 0o600)
    return sha256(data).hexdigest()


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def generation_lookup_url(generation_id: str) -> str:
    if not generation_id.strip():
        raise OpenRouterBillingError("generation id is empty")
    return f"{OPENROUTER_GENERATION_URL}?{urlencode({'id': generation_id})}"


def key_usage_delta(*, pre_usage: Decimal, post_usage: Decimal) -> Decimal:
    delta = post_usage - pre_usage
    if delta < 0:
        raise OpenRouterBillingError("post-run key usage is lower than the pre-run snapshot")
    return delta


class UrllibTransport:
    """Stdlib HTTPS transport. Authorization is request-only and never returned."""

    def send(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> HttpResponse:
        _assert_openrouter_url(url)
        request = Request(url, data=body, method=method)
        for header_name, header_value in headers.items():
            request.add_header(header_name, header_value)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return HttpResponse(status_code=int(response.status), body=bytes(response.read()))
        except HTTPError as exc:
            return HttpResponse(status_code=int(exc.code), body=bytes(exc.read()))
        except URLError as exc:
            raise OpenRouterTransportError("openrouter_network_error") from exc


def parse_object_payload(raw: bytes) -> dict[str, Any]:
    parsed = load_json_preserving_decimals(raw)
    if not isinstance(parsed, dict):
        raise OpenRouterBillingError("OpenRouter payload must be a JSON object")
    return cast(dict[str, Any], parsed)


def _assert_openrouter_url(url: str) -> None:
    allowed = url in {OPENROUTER_KEY_URL, OPENROUTER_CHAT_COMPLETIONS_URL} or url.startswith(
        f"{OPENROUTER_GENERATION_URL}?"
    )
    if not allowed:
        raise OpenRouterBillingError("refusing to call a URL outside the frozen OpenRouter evidence surface")


def _require_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenRouterBillingError(f"{field} must be an integer")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _copy_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _copy_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_jsonable(item) for item in value]
    if isinstance(value, Decimal):
        return format_decimal(value)
    return value


def _redact_secret_strings(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact_secret_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secret_strings(item) for item in value]
    if isinstance(value, str) and any(marker in value for marker in _SECRET_MARKERS):
        return "[REDACTED_SECRET]"
    return value


def utc_now_iso(clock: Clock) -> str:
    return clock.now_utc().astimezone(UTC).isoformat()


def require_git_execution_boundary(spec_path: Path) -> str:
    from .experiment_provenance import (
        require_clean_git_worktree,
        require_committed_clean_experiment_spec,
    )

    snapshot = require_committed_clean_experiment_spec(spec_path)
    head = require_clean_git_worktree(spec_path)
    if snapshot.source_commit_sha != head:
        raise ExperimentProvenanceError("HEAD does not match the committed experiment specification snapshot")
    return head
