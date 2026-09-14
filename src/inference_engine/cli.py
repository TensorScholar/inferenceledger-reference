from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from .domain.models.request import InferenceRequest, ModelParameters
from .infrastructure.models.errors import ProviderError, classify_openai_error
from .infrastructure.models.openai_backend import OpenAIBackend, RetryPolicy
from .infrastructure.telemetry.request_log import JsonlRequestLog, RequestTrace


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one OpenAI-compatible inference smoke request")
    parser.add_argument("--provider", choices=["openai"], default="openai")
    parser.add_argument("--execution-provider", default=None)
    parser.add_argument("--pricing-provider", default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-path", default="reports/request-ledger.jsonl")
    args = parser.parse_args()
    return asyncio.run(_run_smoke(args))


async def _run_smoke(args: argparse.Namespace) -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required for the openai-compatible smoke transport")
        return 2

    request = InferenceRequest(
        prompt=args.prompt,
        parameters=ModelParameters(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    )
    backend = OpenAIBackend(
        api_key=api_key,
        model_name=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        retry_policy=RetryPolicy(max_attempts=2),
        provider_name=getattr(args, "execution_provider", None),
        pricing_provider=getattr(args, "pricing_provider", None),
    )
    request_log = JsonlRequestLog(Path(args.log_path))
    started = perf_counter()

    try:
        response = await backend.infer(request)
    except ProviderError as exc:
        latency_ms = int((perf_counter() - started) * 1000)
        request_log.append(
            RequestTrace.from_error(
                request_id=request.id,
                provider=backend.provider_name,
                model=args.model,
                latency_ms=latency_ms,
                error=exc,
            )
        )
        print(f"provider_error={exc.error_type.value} latency_ms={latency_ms}")
        return 1
    except Exception as exc:
        latency_ms = int((perf_counter() - started) * 1000)
        provider_error = replace(classify_openai_error(exc), provider=backend.provider_name)
        request_log.append(
            RequestTrace.from_error(
                request_id=request.id,
                provider=backend.provider_name,
                model=args.model,
                latency_ms=latency_ms,
                error=provider_error,
            )
        )
        print(f"provider_error={provider_error.error_type.value} latency_ms={latency_ms}")
        return 1

    request_log.append(RequestTrace.from_response(provider=backend.provider_name, response=response))
    print(
        " ".join(
            [
                "ok=true",
                f"provider={backend.provider_name}",
                f"model={response.model_used}",
                f"latency_ms={response.latency_ms}",
                f"prompt_tokens={response.usage.prompt_tokens}",
                f"completion_tokens={response.usage.completion_tokens}",
                f"final_attempt_cost_usd={_format_optional_cost(response.usage.cost_usd)}",
                f"log_path={args.log_path}",
            ]
        )
    )
    return 0


def _format_optional_cost(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.8f}"


if __name__ == "__main__":
    raise SystemExit(main())
