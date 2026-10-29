#!/usr/bin/env python3
"""Measure chat pipeline stage timings against a running backend (no document text logged)."""

from __future__ import annotations

import argparse
import json
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="POST /chat and print timings_ms.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--message", default="What skills are in my resume?")
    args = parser.parse_args()

    payload = {"message": args.message, "history": []}
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{args.base_url.rstrip('/')}/chat", json=payload)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    timings = body.get("timings_ms", {})
    print(json.dumps({"intent": body.get("intent"), "timings_ms": timings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
