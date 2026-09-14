from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from inference_engine.benchmarking.evidence_completeness import (
    assess_decision_evidence_completeness,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="audit_decision_completeness",
        description=(
            "Audit whether a migration decision artifact explicitly represents every required "
            "evidence surface."
        ),
    )
    parser.add_argument("--decision-json", required=True)
    parser.add_argument("--output", default=None)
    return _run(parser.parse_args())


def _run(args: argparse.Namespace) -> int:
    decision_path = Path(args.decision_json)
    raw: Any = json.loads(decision_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("decision JSON must contain an object")

    result = assess_decision_evidence_completeness(raw)
    payload = asdict(result)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        " ".join(
            [
                f"contract={result.contract_version}",
                f"satisfied={str(result.contract_satisfied).lower()}",
                f"present={result.present_count}",
                f"explicitly_unavailable={result.explicitly_unavailable_count}",
                f"missing={result.missing_count}",
                f"missing_checks={','.join(result.missing_checks) or '-'}",
            ]
        )
    )
    return 0 if result.contract_satisfied else 2


if __name__ == "__main__":
    raise SystemExit(main())
