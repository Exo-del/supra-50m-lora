"""Canonical chat markup shared by prepare_data.py and train.py.

Why plain-text markers instead of real special tokens:
the base tokenizer has a fixed 32k byte-level BPE vocab and
tie_word_embeddings=True. Adding tokens means resize_token_embeddings,
which creates new random embedding rows that LoRA cannot train (LoRA
only touches attention/MLP projections). With ~40 optimizer steps on a
50M model those rows would stay noise and corrupt generation. So we
standardise on textual markers that the existing vocab already encodes,
and strip every foreign control token from the sources.
"""

from __future__ import annotations

import json
import re

M = {
    "system": "### System:\n",
    "user": "### User:\n",
    "thinking": "### Thinking:\n",
    "response": "### Response:\n",
    "tool_call": "### Tool call:\n",
    "tool_result": "### Tool result:\n",
}

# Control tokens from other model families / templates we must not inherit.
_ANGLE = re.compile(r"<\|[^|>\n]{1,40}\|>")
_TAGS = re.compile(
    r"</?(?:think|thinking|reasoning|im_start|im_end|s|pad|mask|unk|"
    r"begin_of_text|end_of_text|start_header_id|end_header_id|assistant|user|system)>",
    re.IGNORECASE,
)
_SQUARE = re.compile(r"\[/?INST\]|<<SYS>>|</?s>")
_WS = re.compile(r"[ \t]+\n")
_MULTI = re.compile(r"\n{3,}")


def sanitize(text: str | None) -> str:
    """Strip foreign control tokens and normalise whitespace."""
    if not text:
        return ""
    text = _ANGLE.sub("", text)
    text = _TAGS.sub("", text)
    text = _SQUARE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\u00a0", " ")
    text = _MULTI.sub("\n\n", _WS.sub("\n", text))
    return text.strip()


def clip(text: str, limit: int) -> str:
    """Trim to `limit` chars on a sentence/line boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ("\n", ". ", "! ", "? ", ", "):
        idx = cut.rfind(sep)
        if idx >= limit * 0.5:
            return cut[: idx + len(sep)].rstrip() + (" …" if sep in (". ", "! ", "? ") else "")
    return cut.rstrip() + " …"


def tool_calls_to_text(tool_calls: list | None) -> str:
    """Serialise OpenAI-style tool_calls into a readable canonical block."""
    if not tool_calls:
        return ""
    blocks = []
    for call in tool_calls:
        fn = (call or {}).get("function") or {}
        name = fn.get("name") or "unknown"
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            args = {"_raw": raw}
        blocks.append(f"{name}({json.dumps(args, ensure_ascii=False, sort_keys=True)})")
    return "\n".join(blocks)


def render(messages: list[dict]) -> list[dict]:
    """Canonical messages -> ordered segments.

    Segment: {"text": str, "train": bool, "eos": bool}
      train=True  -> loss is computed on these tokens (assistant output)
      train=False -> context only, masked with -100
      eos=True    -> append </s> after this segment, supervised
    """
    segs: list[dict] = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        content = sanitize(msg.get("content"))
        thinking = sanitize(msg.get("thinking") or msg.get("reasoning"))
        calls = tool_calls_to_text(msg.get("tool_calls"))

        if role == "system":
            if content:
                segs.append({"text": M["system"] + content + "\n", "train": False, "eos": False})

        elif role in ("user", "human"):
            if content:
                segs.append({"text": M["user"] + content + "\n", "train": False, "eos": False})

        elif role in ("tool", "function"):
            body = content or sanitize(msg.get("name"))
            segs.append({"text": M["tool_result"] + body + "\n", "train": False, "eos": False})

        elif role == "assistant":
            text = ""
            if thinking:
                text += M["thinking"] + thinking + "\n"
            text += M["response"]
            if content:
                text += content
            if calls:
                text += ("\n" if content else "") + M["tool_call"] + calls
            if not text.strip():
                continue
            segs.append({"text": text + "\n", "train": True, "eos": True})

        else:
            if content:
                segs.append({"text": M["user"] + content + "\n", "train": False, "eos": False})

    if not segs or not any(s["train"] for s in segs):
        return []
    return segs


def to_text(messages: list[dict]) -> str:
    """Flatten to plain text (for inspection/export)."""
    out = []
    for s in render(messages):
        out.append(s["text"] + ("</s>" if s["eos"] else ""))
    return "".join(out)


def normalize_record(row: dict) -> list[dict] | None:
    """Accept several row shapes and return canonical messages."""
    if row.get("messages"):
        msgs = []
        for m in row["messages"]:
            reasoning = m.get("reasoning") or m.get("thinking")
            msgs.append(
                {
                    "role": m.get("role"),
                    "content": m.get("content"),
                    "thinking": reasoning,
                    "tool_calls": m.get("tool_calls"),
                    "name": m.get("name"),
                }
            )
        return msgs

    if row.get("prompt") and row.get("response"):
        return [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"], "thinking": row.get("thinking")},
        ]
    return None
