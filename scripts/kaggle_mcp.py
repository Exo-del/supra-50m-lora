#!/usr/bin/env python3
"""Push kaggle/run_kaggle.py to Kaggle as a GPU script kernel and run it.

Self-contained: Kaggle MCP over streamable HTTP with a KGAT bearer token.
"""
import json
import pathlib
import sys
import urllib.request

URL = "https://www.kaggle.com/mcp"
TOKEN = "KGAT_7ac3b44e857b4ae9b6eb211c80aa6e18"
REPO = pathlib.Path("/home/louhmiwaslost/Desktop/Workspace/small_agent_50M")
SRC = REPO / "kaggle" / "run_kaggle.py"
SLUG = "supra50m-v5-lora"


def _post(body, sid=None):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    if sid:
        req.add_header("Mcp-Session-Id", sid)
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read().decode()
        out = None
        for line in raw.splitlines():
            if line.startswith("data: "):
                out = json.loads(line[6:])
        if out is None and raw.strip():
            out = json.loads(raw)
        return out, r.headers.get("Mcp-Session-Id")


def call(name, args):
    init, sid = _post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                  "clientInfo": {"name": "opencode", "version": "1"}}})
    try:
        _post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, sid)
    except Exception:
        pass
    return _post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                  "params": {"name": name, "arguments": args}}, sid)[0]


def main() -> int:
    text = SRC.read_text()
    print(f"[*] script: {SRC} ({len(text)} bytes)")
    req = {
        "slug": SLUG, "slugNullable": SLUG, "hasSlug": True,
        "newTitle": "Supra-50M v5 LoRA: compact json schema 2048 ctx",
        "newTitleNullable": "Supra-50M v5 LoRA: compact json schema 2048 ctx",
        "hasNewTitle": True,
        "text": text, "textNullable": text, "hasText": True,
        "language": "python", "languageNullable": "python", "hasLanguage": True,
        "kernelType": "script", "kernelTypeNullable": "script", "hasKernelType": True,
        "isPrivate": True, "isPrivateNullable": True, "hasIsPrivate": True,
        "enableGpu": True, "enableGpuNullable": True, "hasEnableGpu": True,
        "enableTpu": False, "enableTpuNullable": False, "hasEnableTpu": True,
        "enableInternet": True, "enableInternetNullable": True, "hasEnableInternet": True,
        "kernelExecutionType": "SaveAndRunAll",
        "kernelExecutionTypeNullable": "SaveAndRunAll",
        "hasKernelExecutionType": True,
        "datasetDataSourcesSetter": [],
        "kernelDataSourcesSetter": [],
        "competitionDataSourcesSetter": [],
        "modelDataSourcesSetter": [],
        "categoryIdsSetter": [],
        "sessionTimeoutSeconds": 43200, "sessionTimeoutSecondsNullable": 43200,
        "hasSessionTimeoutSeconds": True,
        "priority": 1, "priorityNullable": 1, "hasPriority": True,
    }
    out = call("save_notebook", {"request": req})
    res = out.get("result", {})
    if res.get("isError"):
        print("ERROR:", res.get("content"))
        return 1
    payload = res.get("content")
    if isinstance(payload, list):
        for part in payload:
            print(part.get("text", "")[:2000])
    else:
        print(str(payload)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
