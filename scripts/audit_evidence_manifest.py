from __future__ import annotations

import argparse
from pathlib import Path

from inference_engine.benchmarking.evidence_manifest import load_manifest, verify_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="audit_evidence_manifest",
        description="Verify committed/sanitized pilot bundle bytes against the deterministic manifest.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bundle-root", required=True)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    result = verify_manifest(manifest, root=Path(args.bundle_root))
    print(
        " ".join(
            [
                f"valid={str(result.valid).lower()}",
                f"checked_local={result.checked_local_count}",
                f"private_references={result.private_reference_count}",
                f"unavailable={result.unavailable_count}",
                f"errors={len(result.errors)}",
            ]
        )
    )
    for error in result.errors:
        print(f"error={error}")
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
