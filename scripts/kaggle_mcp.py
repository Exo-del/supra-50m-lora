#!/usr/bin/env python3
"""Push kaggle/run_kaggle.py to Kaggle as a GPU script kernel and run it.

Usage: python scripts/kaggle_mcp.py "<Title With No Punctuation>"

Notes
-----
* Do not send ``slug``: the server derives it from ``newTitle``. Sending an
  explicit slug (or ``null``) fails with ``Invalid slug: '...'``.
* Titles must be unique per user, so bump the version in the title each run.
* A KGAT bearer token is enough for POST /api/v1/kernels/push.
"""
import json
import pathlib
import sys
import urllib.request

PUSH = "https://www.kaggle.com/api/v1/kernels/push"
TOKEN = "KGAT_7ac3b44e857b4ae9b6eb211c80aa6e18"
REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "kaggle" / "run_kaggle.py"

TITLE = sys.argv[1] if len(sys.argv) > 1 else "Supra 50M v5 LoRA compact json schema 2048 ctx"


def push(body: dict) -> dict:
    req = urllib.request.Request(PUSH, data=json.dumps(body).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"httpStatus": e.code, "body": e.read().decode()[:600]}


def main() -> int:
    text = SRC.read_text()
    print(f"[*] script: {SRC} ({len(text)} bytes)")
    print(f"[*] title : {TITLE}")
    body = {
        "newTitle": TITLE,
        "text": text,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": True,
        "enableTpu": False,
        "enableInternet": True,
        "kernelExecutionType": "SaveAndRunAll",
        "sessionTimeoutSeconds": 43200,
        "machineShape": "",
        "dockerImage": "",
        "datasetDataSources": [],
        "kernelDataSources": [],
        "competitionDataSources": [],
        "modelDataSources": [],
        "categoryIds": [],
        "priority": 1,
    }
    out = push(body)
    print(json.dumps(out, indent=1)[:1500])
    return 0 if not out.get("hasError") and not out.get("httpStatus") else 1


if __name__ == "__main__":
    sys.exit(main())
