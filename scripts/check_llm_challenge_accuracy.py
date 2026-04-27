from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.parse import urlparse
from dataclasses import asdict, dataclass
from typing import Any

import requests


@dataclass
class CheckRecord:
    index: int
    prediction: str
    expected: str
    expected_match: bool
    live_result: bool | None
    live_reason: str
    endpoint_error: str
    is_correct: bool | None


DEFAULT_CHALLENGE_EXAMPLES: list[tuple[str, str, bool]] = [
    ("irish terrier", "dog", True),
    ("tabby", "cat", True),
    ("persian cat", "cat", True),
    ("goldfish", "fish", True),
    ("alligator lizard", "reptile", True),
    ("snail", "mollusk", True),
    ("black and gold garden spider", "arachnid", True),
    ("american lobster", "crustacean", True),
    ("bullfrog", "amphibian", True),
    ("sorrel", "equine", True),
    ("street sign", "porcine", False),
    ("pillow", "spiny lobster", False),
]


def _build_records(extra_examples: list[str]) -> list[CheckRecord]:
    examples = list(DEFAULT_CHALLENGE_EXAMPLES)
    for raw in extra_examples:
        # Format: prediction|target|expected_match
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) != 3:
            raise ValueError(f"Invalid --example '{raw}'. Use prediction|target|expected_match")
        prediction, expected, expected_match_raw = parts
        expected_match = expected_match_raw.lower() in {"true", "1", "yes", "y"}
        examples.append((prediction, expected, expected_match))
    return [
        CheckRecord(
            index=i + 1,
            prediction=prediction,
            expected=expected,
            expected_match=expected_match,
            live_result=None,
            live_reason="",
            endpoint_error="",
            is_correct=None,
        )
        for i, (prediction, expected, expected_match) in enumerate(examples)
    ]


def _verify_live(
    records: list[CheckRecord],
    endpoint: str,
    llm_model: str,
    timeout_seconds: float,
) -> None:
    for rec in records:
        payload = {
            "prediction": rec.prediction,
            "target_label": rec.expected,
            "llm_model": llm_model,
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            data: Any = response.json()
            if isinstance(data, dict):
                value = data.get("is_match")
                reason = data.get("reason", "")
                rec.live_reason = str(reason) if reason is not None else ""
                if isinstance(value, bool):
                    rec.live_result = value
                elif isinstance(value, str):
                    rec.live_result = value.strip().lower() in {"true", "1", "yes"}
                else:
                    rec.endpoint_error = "missing/invalid is_match in response"
            else:
                rec.endpoint_error = "non-object JSON response"
        except Exception as exc:
            rec.endpoint_error = str(exc)
        if rec.live_result is not None:
            rec.is_correct = rec.live_result == rec.expected_match


def _summarize(records: list[CheckRecord]) -> dict[str, Any]:
    total = len(records)
    live_known = [r for r in records if r.live_result is not None]
    correct = [r for r in live_known if r.is_correct is True]

    return {
        "total_examples": total,
        "live_result_available": len(live_known),
        "accuracy": (len(correct) / len(live_known)) if live_known else None,
        "pass_count": len(correct),
        "fail_count": len(live_known) - len(correct),
        "endpoint_error_count": sum(1 for r in records if r.endpoint_error),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check llm challenge verification accuracy using manual examples."
    )
    parser.add_argument(
        "--llm-endpoint",
        default="http://127.0.0.1:8081/verify-label",
        help="Local LLM verification endpoint URL.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("PERTURB_LLM_ENDPOINT_MODEL", "Qwen2.5-1.5B-Instruct"),
        help="Model hint passed to verification endpoint.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="HTTP timeout for each live verification request.",
    )
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        help="Add extra example in format prediction|target|expected_match",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path for full JSON report.",
    )
    args = parser.parse_args()

    parsed = urlparse(args.llm_endpoint)
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"127.0.0.1", "localhost"}:
        print(
            f"llm endpoint must be local (127.0.0.1 or localhost), got: {args.llm_endpoint}",
            file=sys.stderr,
        )
        return 1

    try:
        records = _build_records(args.example)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    _verify_live(
        records=records,
        endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        timeout_seconds=args.timeout_seconds,
    )
    summary = _summarize(records)

    print("=== LLM Manual Challenge Accuracy Summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Per-Example Results ===")
    for rec in records:
        print(
            f"[{rec.index}] pred='{rec.prediction}' expected='{rec.expected}' expected_match={rec.expected_match} "
            f"live={rec.live_result} correct={rec.is_correct} "
            f"error='{rec.endpoint_error}' reason='{rec.live_reason}'"
        )

    if args.output_json:
        payload = {
            "summary": summary,
            "records": [asdict(r) for r in records],
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSaved full report to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

