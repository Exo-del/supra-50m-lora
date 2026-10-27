#!/usr/bin/env python3
"""Fetch a Kaggle kernel's log and output files via the v1 API.

    python scripts/fetch_kaggle_run.py --slug supra-50m-v4-lora-english-agentic-reasoning

Reads the bearer token from ~/.config/opencode/opencode.jsonc (kaggle MCP entry)
or from KAGGLE_BEARER. Output goes to runs/<slug>/.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request

API = "https://www.kaggle.com/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent.parent


def token() -> str:
    env = os.environ.get("KAGGLE_BEARER")
    if env:
        return env
    cfg = pathlib.Path.home() / ".config/opencode/opencode.jsonc"
    text = cfg.read_text()
    marker = '"Authorization": "Bearer '
    idx = text.find(marker, text.find('"kaggle"'))
    if idx < 0:
        raise SystemExit("[!] no token: set KAGGLE_BEARER")
    return text[idx + len(marker): text.index('"', idx + len(marker))]


def get(path: str, tok: str):
    req = urllib.request.Request(f"{API}/{path}")
    req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="alouhmiwaslost")
    ap.add_argument("--slug", default="supra-50m-v4-lora-english-agentic-reasoning")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    tok = token()
    out = args.out or ROOT / "runs" / args.slug
    out.mkdir(parents=True, exist_ok=True)

    status = json.loads(get(f"kernels/status?userName={args.user}&kernelSlug={args.slug}", tok))
    print(f"[*] status: {status.get('status')} {status.get('failureMessage') or ''}".rstrip())

    info = json.loads(get(f"kernels/output?userName={args.user}&kernelSlug={args.slug}", tok))
    log = info.get("log") or ""
    if log:
        (out / "kernel.log").write_text(log)
        print(f"[+] log -> {out / 'kernel.log'} ({len(log):,} chars)")
    else:
        print("[i] log empty (script kernels only publish it near the end)")

    files = info.get("files") or []
    print(f"[*] output files: {len(files)}")
    for f in files:
        name = f.get("name") or f.get("fileName")
        size = f.get("totalBytes") or f.get("size") or 0
        print(f"    {name}  {size:,} bytes" if isinstance(size, int) else f"    {name}")
        if not name or args.list_only:
            continue
        data = get(f"kernels/output/{urllib.parse.quote(name)}?userName={args.user}&kernelSlug={args.slug}", tok)
        dest = out / pathlib.Path(name).name
        dest.write_bytes(data)
        print(f"      -> {dest} ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
