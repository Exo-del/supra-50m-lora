"""Build an English-only mixed SFT set for Supra-50M-Base.

Goal of the mix
    (a) multi-turn agentic tool calling,
    (b) structured reasoning in the DeepSeek "thinking then answer" shape,
    (c) a logic floor from math and code traces,
    (d) a no-tool negative class so the model does not call a function for
        every prompt.

Sources (all English)
    agent-deepseek  zake7749/deepseek-v4-pro-agent-tool-calling-trajectory
                    multi-turn tool-calling trajectories with per-turn
                    reasoning, 5 domains, heavily duplicated (284 unique ids
                    behind 2510 rows) -> dedupe by id, then slice by turn.
    agent-glaive    glaiveai/glaive-function-calling-v2
                    <functioncall> / FUNCTION RESPONSE traces: real tool
                    results, so the model sees cause and effect, not just
                    call syntax.
    math-gsm8k      openai/gsm8k
                    short clean CoT, `<<..=..>>` calculator markers stripped.
    math-instruct   TIGER-Lab/MathInstruct
                    short CoT / program-of-thought answers across maths.
    think-code      open-thoughts/OpenThoughts-114k
                    R1-style Thought/Solution traces, FILTERED TO SHORT ONES.
    general         databricks/databricks-dolly-15k
                    no-tool answers (negative class), context-free only.
    seed            data/seed.jsonl (local anchors, hand written)

Dropped on purpose
    Davizig10jojo/Kimi-K3-And-DeepSeek-V4-Pro-0813-Distillation-in-PT-BR
    was in v1-v3 of this mix. It is Portuguese, the target is an English
    agentic/reasoning adapter, and 58 rows buy ~18% loss reduction on a slice
    that is 3% of the data. Removed in v4; see README "What changed in v4".

Why short traces are filtered, not clipped
    Truncating a chain-of-thought mid-way teaches the model to ramble and
    never conclude. For reasoning sources we keep only rows whose *whole*
    trace fits the sequence budget. Agentic rows are different: they are
    multi-turn, so they are sliced at turn boundaries and still end on a
    complete assistant turn.

Usage
    python prepare_data.py --profile kaggle --target-tokens 6000000
    python prepare_data.py --profile cpu --target-tokens 400000 --only seed,general
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics as stats
from collections import Counter
from pathlib import Path

import canon

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "mix.jsonl"
SEED_FILE = ROOT / "data" / "seed.jsonl"

DS_AGENT = "zake7749/deepseek-v4-pro-agent-tool-calling-trajectory"
DS_GLAIVE = "glaiveai/glaive-function-calling-v2"
DS_GSM8K = "openai/gsm8k"
DS_MATHIN = "TIGER-Lab/MathInstruct"
DS_THINK = "open-thoughts/OpenThoughts-114k"
DS_GENERAL = "databricks/databricks-dolly-15k"

# Token share per source. Renormalised over whichever sources are enabled.
WEIGHTS = {
    "agent-deepseek": 0.26,
    "agent-glaive": 0.18,
    "math-gsm8k": 0.13,
    "math-instruct": 0.11,
    "think-code": 0.16,
    "general": 0.14,
    "seed": 0.02,
}

# Clip budgets in chars per profile.
#   cpu   : ~330 tokens/row, fits max_seq 384 on a 12s/step CPU
#   kaggle: full 1024 ctx, light clipping only
PROFILES = {
    "cpu": dict(think=300, answer=460, prompt=180, sys_pt=160,
                sys_agent=200, msg=240, reason=200, tool=160, budget=850,
                slice_ctx=850, max_tokens=384),
    "kaggle": dict(think=900, answer=1600, prompt=320, sys_pt=420,
                   sys_agent=520, msg=900, reason=700, tool=520, budget=2600,
                   slice_ctx=1700, max_tokens=1024),
}
B = dict(PROFILES["kaggle"])


# --------------------------------------------------------------- helpers
def slice_turns(msgs: list[dict], k: int, ctx_chars: int) -> list[list[dict]]:
    """Emit up to k examples that each *end* on a different assistant turn.

    Tool-calling trajectories are multi-turn; training only on the final
    turn wastes the earlier ones. Each slice keeps a bounded context
    window so it still fits the sequence budget.
    """
    asst = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if k <= 1 or len(asst) <= 1:
        return [msgs]

    out = []
    for i in asst[-k:]:
        ctx, spent = [], 0
        for j in range(i - 1, -1, -1):
            m = msgs[j]
            cost = len(m.get("content") or "") + len(m.get("thinking") or "")
            if spent + cost > ctx_chars:
                break
            ctx.append(m)
            spent += cost
        ctx.reverse()
        # keep the system prompt if it fell outside the window
        if msgs and msgs[0].get("role") == "system" and (not ctx or ctx[0].get("role") != "system"):
            ctx = [msgs[0]] + ctx
        out.append(ctx + [msgs[i]])
    return out


def _digest(msgs: list[dict]) -> str:
    txt = " ".join(canon.to_text(msgs).split())
    return hashlib.sha1(txt.encode()).hexdigest()


def _stratified(rows: list[dict], limit: int, key: str, per_group: int, rng: random.Random) -> list[dict]:
    """Round-robin over groups so every domain/style gets representation."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(key) or "na"), []).append(row)
    for items in buckets.values():
        rng.shuffle(items)

    per_group = max(per_group, -(-limit // max(1, len(buckets))))

    picked, names = [], list(buckets)
    rng.shuffle(names)
    while len(picked) < limit and names:
        for name in list(names):
            bucket = buckets[name]
            taken = sum(1 for p in picked if str(p.get(key) or "na") == name)
            if not bucket or taken >= per_group:
                names.remove(name)
                continue
            picked.append(bucket.pop())
            if len(picked) >= limit:
                break
    return picked


def _row(msgs: list[dict], source: str, domain: str) -> dict:
    return {"messages": msgs, "source": source, "lang": "en", "domain": domain}


# --------------------------------------------------------------- sources
def from_agent_deepseek(cap: int, rng: random.Random, slice_k: int = 4) -> list[dict]:
    """English multi-turn trajectories: reasoning + tool calls + tool results."""
    from datasets import load_dataset

    ds = load_dataset(DS_AGENT, split="train")
    rows, seen_ids = [], set()
    for row in ds:
        if (row.get("score") or 0) < 0.66:
            continue
        if (row.get("language") or "en") != "en":
            continue
        rid = row.get("id")
        if rid in seen_ids:  # 2510 rows / 284 ids: clones add nothing
            continue
        seen_ids.add(rid)
        if (row.get("num_turns") or 0) < 2:
            continue

        msgs = []
        for msg in row["messages"]:
            role = (msg.get("role") or "").lower()
            content = canon.sanitize(msg.get("content"))
            reasoning = canon.sanitize(msg.get("reasoning"))

            if role == "system":
                entry = {"role": "system", "content": canon.clip(content, B["sys_agent"])}
            elif role == "user":
                entry = {"role": "user", "content": canon.clip(content, B["msg"])}
            elif role == "assistant":
                entry = {
                    "role": "assistant",
                    "content": canon.clip(content, B["msg"]),
                    "thinking": canon.clip(reasoning, B["reason"]) if reasoning else "",
                    "tool_calls": msg.get("tool_calls"),
                }
                if not entry["content"] and not entry["tool_calls"]:
                    continue
            elif role in ("tool", "function"):
                entry = {"role": "tool", "content": canon.clip(content, B["tool"]), "name": msg.get("name")}
            else:
                continue
            msgs.append(entry)

        while msgs and msgs[-1].get("role") != "assistant":
            msgs.pop()
        if not any(m.get("role") == "assistant" for m in msgs):
            continue

        for variant in slice_turns(msgs, slice_k, B["slice_ctx"]):
            rows.append(_row(variant, "agent-deepseek", row.get("domain") or "na"))
        if len(rows) >= cap:
            break
    return rows[:cap]


_FN_TAG = "<functioncall>"
_GLAIVE_ROLE = re.compile(r"^(SYSTEM|USER|ASSISTANT|FUNCTION RESPONSE):", re.M)


def _extract_json_object(text: str, start: int) -> tuple[str | None, int]:
    """Scan one balanced JSON object from text[start]; quote-aware.

    glaive wraps calls as `<functioncall> {...}` and often omits the closing
    tag, while `arguments` is a *nested JSON string*, so a non-greedy regex
    stops inside the wrong brace. Walk the braces instead.
    """
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1], i + 1
    return None, len(text)


def _repair_call(blob: str) -> dict | None:
    """Recover a call from glaive's not-quite-JSON arguments.

    The dataset writes arguments as a single-quoted JSON string
    ('{"country": "United States"}'), which json.loads rejects. Pull the
    name and the raw argument payload out by hand; the renderer accepts
    either a JSON string or a dict.
    """
    m = re.search(r'"name"\s*:\s*"([^"]+)"', blob)
    if not m:
        return None
    name = m.group(1)
    a = re.search(r'"arguments"\s*:\s*(.+?)\s*\}\s*$', blob, re.S)
    raw = a.group(1).strip() if a else "{}"
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1]
    try:
        json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raw = raw.strip() or "{}"
    return {"name": name, "arguments": raw}


def _split_functioncalls(body: str) -> tuple[list[dict], str]:
    """Pull tool calls out of a glaive assistant block, return (calls, text)."""
    calls, text_parts, idx = [], [], 0
    while True:
        j = body.find(_FN_TAG, idx)
        if j < 0:
            text_parts.append(body[idx:])
            break
        text_parts.append(body[idx:j])
        k = body.find("{", j)
        if k < 0:
            idx = j + len(_FN_TAG)
            continue
        blob, end = _extract_json_object(body, k)
        obj = None
        if blob:
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                obj = _repair_call(blob)
        if obj:
            calls.append({"function": {"name": obj.get("name"), "arguments": obj.get("arguments")}})
        idx = end + (len("</functioncall>") if body.startswith("</functioncall>", end) else 0)
    text = "".join(text_parts).replace("</functioncall>", "").strip()
    return calls, text



def from_agent_glaive(cap: int, rng: random.Random, slice_k: int = 3) -> list[dict]:
    """glaive-v2: call -> FUNCTION RESPONSE -> answer. Real tool results."""
    from datasets import load_dataset

    ds = load_dataset(DS_GLAIVE, split="train")
    rows = []
    for row in ds:
        chat = row.get("chat") or ""
        if "<functioncall>" not in chat and "FUNCTION RESPONSE" not in chat:
            continue

        msgs = []
        sys_txt = canon.sanitize((row.get("system") or "").replace("SYSTEM:", "", 1))
        if sys_txt:
            msgs.append({"role": "system", "content": canon.clip(sys_txt, B["sys_agent"])})

        # split the flat transcript into role blocks
        parts = _GLAIVE_ROLE.split(chat)
        for i in range(1, len(parts) - 1, 2):
            role, body = parts[i], canon.sanitize(parts[i + 1])
            if not body:
                continue
            if role == "USER":
                msgs.append({"role": "user", "content": canon.clip(body, B["msg"])})
            elif role == "FUNCTION RESPONSE":
                last = next((m for m in reversed(msgs) if m.get("tool_calls")), None)
                name = ((last or {}).get("tool_calls") or [{}])[0].get("function", {}).get("name")
                msgs.append({"role": "tool", "content": canon.clip(body, B["tool"]), "name": name})
            elif role == "ASSISTANT":
                calls, text = _split_functioncalls(body)
                entry = {"role": "assistant", "content": canon.clip(text, B["answer"])}
                if calls:
                    entry["tool_calls"] = calls
                if not entry["content"] and not calls:
                    continue
                msgs.append(entry)
            elif role == "SYSTEM":
                msgs.append({"role": "system", "content": canon.clip(body, B["sys_agent"])})

        while msgs and msgs[-1].get("role") != "assistant":
            msgs.pop()
        if not any(m.get("role") == "assistant" for m in msgs):
            continue

        domain = "tool-multi" if any(m.get("role") == "tool" for m in msgs) else "no-tool"
        for variant in slice_turns(msgs, slice_k, B["slice_ctx"]):
            rows.append(_row(variant, "agent-glaive", domain))
        if len(rows) >= cap:
            break
    return rows[:cap]


def from_math_gsm8k(cap: int, rng: random.Random) -> list[dict]:
    """GSM8K: short grade-school CoT. Thinking = steps, answer = final line."""
    from datasets import load_dataset

    ds = load_dataset(DS_GSM8K, "main", split="train")
    rows = []
    for row in ds:
        q = canon.sanitize(row.get("question"))
        ans = canon.sanitize(canon.strip_calc_markers(row.get("answer") or ""))
        if not q or "####" not in ans:
            continue
        steps, _, final = ans.rpartition("####")
        steps, final = steps.strip(), final.strip()
        if len(steps) < 60:
            continue
        msgs = [
            {"role": "user", "content": canon.clip(q, B["prompt"])},
            {"role": "assistant", "thinking": canon.clip(steps, B["think"]), "content": final},
        ]
        rows.append(_row(msgs, "math-gsm8k", "arithmetic"))
        if len(rows) >= cap:
            break
    return rows[:cap]


def from_math_instruct(cap: int, rng: random.Random) -> list[dict]:
    """MathInstruct: short CoT / program-of-thought across maths."""
    from datasets import load_dataset

    ds = load_dataset(DS_MATHIN, split="train")
    rows = []
    for row in ds:
        q = canon.sanitize(row.get("instruction"))
        out = canon.sanitize(row.get("output"))
        if not q or len(out) < 80:
            continue
        low = out.lower()
        idx = low.rfind("the answer is")
        if idx > 40:
            thinking = out[:idx].strip().rstrip(".:")
            answer = out[idx + len("the answer is"):].strip(" .:\n")
        else:
            thinking, answer = "", out
        msgs = [
            {"role": "user", "content": canon.clip(q, B["prompt"])},
            {"role": "assistant", "thinking": canon.clip(thinking, B["think"]), "content": canon.clip(answer, B["answer"])},
        ]
        src_file = str(row.get("source") or "math")
        domain = "code" if "code" in src_file.lower() else "math"
        rows.append(_row(msgs, "math-instruct", domain))
        if len(rows) >= cap:
            break
    return rows[:cap]


def from_think_code(cap: int, rng: random.Random, max_chars: int = 3000) -> list[dict]:
    """OpenThoughts-114k: R1-style Thought/Solution, short rows only.

    Long traces are *dropped*, not truncated: a clipped chain-of-thought
    teaches the model to never reach an answer.
    """
    from datasets import load_dataset

    ds = load_dataset(DS_THINK, split="train", streaming=True)
    rows, scanned = [], 0
    for row in ds:
        scanned += 1
        conv = row.get("conversations") or []
        if len(conv) < 2:
            continue
        user = canon.sanitize(conv[0].get("value"))
        raw_asst = conv[1].get("value") or ""
        if not user or len(raw_asst) > max_chars:
            continue

        # Split on the template markers *before* sanitize strips them,
        # otherwise the whole trace lands in the answer and the model never
        # learns the "think, then answer" shape.
        t_raw, a_raw = "", raw_asst
        low_raw = raw_asst.lower()
        for marker in ("<|end_of_thought|>", "<|begin_of_solution|>", "<|end_of_solution|>"):
            pos = low_raw.find(marker)
            if pos >= 0:
                t_raw, a_raw = raw_asst[:pos], raw_asst[pos + len(marker):]
                break
        if not t_raw and "## solution" in low_raw:
            pos = low_raw.index("## solution")
            t_raw, a_raw = raw_asst[:pos], raw_asst[pos + len("## solution"):]

        thinking = canon.sanitize(t_raw)
        asst = canon.sanitize(a_raw)
        if len(asst) < 40:
            continue

        domain = str(row.get("domain") or "")
        is_code = "python function" in user.lower() or "code" in domain.lower()
        if not (is_code or "math" in domain.lower()):
            continue

        msgs = [
            {"role": "user", "content": canon.clip(user, B["prompt"])},
            {"role": "assistant", "thinking": canon.clip(thinking, B["think"]), "content": canon.clip(asst, B["answer"])},
        ]
        rows.append(_row(msgs, "think-code", "code" if is_code else "math-think"))
        if len(rows) >= cap or scanned > cap * 40:
            break
    return rows[:cap]


def from_general(cap: int, rng: random.Random) -> list[dict]:
    """General English instruction data, no tools.

    Serves as the negative class: without it the agent rows teach the model
    to emit a tool call for *every* prompt, including plain questions.
    """
    from datasets import load_dataset

    ds = load_dataset(DS_GENERAL, split="train")
    rows = []
    for row in ds:
        instr = canon.sanitize(row.get("instruction"))
        resp = canon.sanitize(row.get("response"))
        if not instr or len(resp) < 60:
            continue
        if row.get("context"):
            continue  # context-grounded rows are summarisation, not Q&A
        msgs = [
            {"role": "user", "content": canon.clip(instr, 200)},
            {"role": "assistant", "content": canon.clip(resp, 340)},
        ]
        rows.append(_row(msgs, "general", row.get("category") or "general"))
        if len(rows) >= cap:
            break
    return rows[:cap]


def from_seed(rng: random.Random) -> list[dict]:
    """Local seed set: short, generic English anchors."""
    if not SEED_FILE.exists():
        return []
    rows = []
    for line in SEED_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = canon.normalize_record(row)
        if msgs:
            rows.append(_row(msgs, "seed", "general"))
    rng.shuffle(rows)
    return rows


BUILDERS = {
    "agent-deepseek": from_agent_deepseek,
    "agent-glaive": from_agent_glaive,
    "math-gsm8k": from_math_gsm8k,
    "math-instruct": from_math_instruct,
    "think-code": from_think_code,
    "general": from_general,
}


# --------------------------------------------------------------- budget
def token_len(msgs: list[dict], tok) -> int:
    n = 0
    for seg in canon.render(msgs):
        n += len(tok(seg["text"], add_special_tokens=False).input_ids) + (1 if seg["eos"] else 0)
    return n


def take_budget(cands: list[dict], budget: int, tok, rng: random.Random, max_len: int) -> tuple[list[dict], dict]:
    """Shuffle, tokenize, keep complete rows until the token budget is met."""
    rng.shuffle(cands)
    kept, stats_ = [], Counter()
    spent = 0
    for row in cands:
        if spent >= budget:
            break
        if row.get("_dupe"):
            stats_["dupe"] += 1
            continue
        n = token_len(row["messages"], tok)
        if n > max_len:
            stats_["overlong"] += 1
            continue
        if not canon.is_english(canon.to_text(row["messages"])):
            stats_["not_english"] += 1
            continue
        row["tokens"] = n
        kept.append(row)
        spent += n
    stats_["tokens"] = spent
    return kept, stats_


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default="kaggle")
    ap.add_argument("--target-tokens", type=int, default=6_000_000,
                    help="total tokens of unique training data to build")
    ap.add_argument("--only", default="", help="comma list of sources to keep")
    ap.add_argument("--skip", default="", help="comma list of sources to skip")
    ap.add_argument("--cap-per-source", type=int, default=20000,
                    help="max candidate rows gathered per source before budgeting")
    ap.add_argument("--slice-turns", type=int, default=4,
                    help="turn-level slices per agentic trajectory")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed-repeat", type=int, default=1)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = profile default")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    B.update(PROFILES[args.profile])
    max_len = args.max_tokens or B["max_tokens"]
    tok = AutoTokenizer.from_pretrained(ROOT / "model")
    rng = random.Random(args.rng_seed)

    enabled = [s for s in WEIGHTS if s not in ("seed",)]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        enabled = [s for s in enabled if s in want]
    if args.skip:
        drop = {s.strip() for s in args.skip.split(",") if s.strip()}
        enabled = [s for s in enabled if s not in drop]
    unknowns = set(args.only.split(",")) | set(args.skip.split(","))
    bad = {u.strip() for u in unknowns if u.strip()} - set(WEIGHTS)
    if bad:
        raise SystemExit(f"[!] unknown source(s): {sorted(bad)}  (known: {sorted(WEIGHTS)})")

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    seed_on = (not only) or ("seed" in only)
    weight_sum = sum(WEIGHTS[s] for s in enabled) + (WEIGHTS["seed"] if seed_on else 0)

    print(f"[*] profile={args.profile} max_tokens={max_len} target={args.target_tokens:,} tokens")
    print(f"[*] sources: {enabled}{' + seed' if seed_on else ''}")

    per_source: dict[str, list[dict]] = {}
    report = {}
    for src in enabled:
        budget = int(args.target_tokens * WEIGHTS[src] / weight_sum)
        print(f"[*] {src}: budget {budget:,} tokens ...", flush=True)
        try:
            if src.startswith("agent-"):
                cands = BUILDERS[src](args.cap_per_source, rng, slice_k=args.slice_turns)
            else:
                cands = BUILDERS[src](args.cap_per_source, rng)
        except Exception as exc:  # network / dataset rename: keep going
            print(f"    [!] source failed: {type(exc).__name__}: {exc}")
            report[src] = {"error": str(exc)}
            continue

        # global dedupe inside the source
        seen, uniq = set(), []
        for row in cands:
            d = _digest(row["messages"])
            if d in seen:
                row["_dupe"] = True
            else:
                seen.add(d)
            uniq.append(row)

        kept, stats_ = take_budget(uniq, budget, tok, rng, max_len)
        per_source[src] = kept
        report[src] = {
            "candidates": len(cands),
            "kept": len(kept),
            "tokens": stats_["tokens"],
            "not_english": stats_["not_english"],
            "overlong": stats_["overlong"],
            "dupes": len(cands) - len(seen),
        }
        print(f"    kept {len(kept)} rows / {stats_['tokens']:,} tok"
              f"  (non-en {stats_['not_english']}, overlong {stats_['overlong']}, dupes {len(cands) - len(seen)})")

    if seed_on:
        seed = from_seed(rng)[: max(16, args.target_tokens // 200000)]
        rows = []
        for r in seed:
            n = token_len(r["messages"], tok)
            if n <= max_len:
                r["tokens"] = n
                rows.append(r)
        for _ in range(max(0, args.seed_repeat - 1)):
            rows.extend({**r} for r in rows)
        per_source["seed"] = rows
        report["seed"] = {"candidates": len(seed), "kept": len(rows),
                          "tokens": sum(r["tokens"] for r in rows), "not_english": 0,
                          "overlong": 0, "dupes": 0}
        print(f"[*] seed: kept {len(rows)} rows")

    records = []
    for src in per_source:
        records.extend(per_source[src])
    if len(records) < 2:
        raise SystemExit("[!] nothing built")
    rng.shuffle(records)

    # deterministic held-out split, stratified by source
    val_rows = []
    n_val = max(1, int(len(records) * args.val_frac))
    by_src: dict[str, list[dict]] = {}
    for r in records:
        by_src.setdefault(r["source"], []).append(r)
    for src, items in by_src.items():
        take = max(1, int(len(items) * args.val_frac))
        val_rows.extend(items[:take])
    val_ids = {id(r) for r in val_rows}
    train_rows = [r for r in records if id(r) not in val_ids]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    train_path = args.out.with_suffix(".train.jsonl")
    val_path = args.out.with_suffix(".val.jsonl")
    for path, rows in ((args.out, records), (train_path, train_rows), (val_path, val_rows)):
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_tok = sum(r["tokens"] for r in records) or 1
    by_src_rows = Counter(r["source"] for r in records)
    by_src_tok = Counter()
    for r in records:
        by_src_tok[r["source"]] += r["tokens"]
    lens = [r["tokens"] for r in records]

    print(f"\n[+] wrote {len(records)} rows -> {args.out}")
    print(f"    split : {len(train_rows)} train / {len(val_rows)} val")
    print(f"    tokens: med {int(stats.median(lens))} / max {max(lens)} / total {sum(lens):,}")
    print(f"    rows  : {dict(by_src_rows)}")
    print("    share : " + ", ".join(f"{k} {100 * v / total_tok:.0f}%" for k, v in by_src_tok.most_common()))
    print(f"    domains: {dict(Counter(r['domain'] for r in records).most_common(10))}")

    summary = {
        "profile": args.profile,
        "target_tokens": args.target_tokens,
        "max_tokens": max_len,
        "rows": len(records),
        "tokens": sum(lens),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "median_tokens": int(stats.median(lens)),
        "max_row_tokens": max(lens),
        "per_source": report,
        "token_share": {k: round(100 * v / total_tok, 1) for k, v in by_src_tok.items()},
        "domains": dict(Counter(r["domain"] for r in records).most_common(20)),
    }
    (args.out.parent / (args.out.stem + ".report.json")).write_text(json.dumps(summary, indent=2))
    print(f"    report: {args.out.parent / (args.out.stem + '.report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
