#!/usr/bin/env python3
"""Call the Go NPV endpoint and print the JSON response."""

from __future__ import annotations

import json
import os
import sys

import httpx

GO_MATH_URL = os.environ.get("GO_MATH_URL", "http://localhost:8081").rstrip("/")

DEFAULT_BODY = {
    "cash_flows": [100.0, 110.0, 121.0, 133.1, 146.41],
    "discount_rate": 0.10,
}


def main() -> int:
    url = f"{GO_MATH_URL}/npv"
    try:
        resp = httpx.post(url, json=DEFAULT_BODY, timeout=10.0)
    except httpx.RequestError as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(resp.text, file=sys.stderr)
        return 1 if resp.status_code >= 400 else 0

    if resp.status_code >= 400:
        print(json.dumps(data, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
