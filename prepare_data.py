"""Sample, clean and mix two Hugging Face datasets into canonical JSONL.

Sources
  1. Davizig10jojo/Kimi-K3-And-DeepSeek-V4-Pro-0813-Distillation-in-PT-BR
     58 rows, pt-BR, long chain-of-thought distillations.
  2. zake7749/deepseek-v4-pro-agent-tool-calling-trajectory
     2510 rows, en, multi-turn tool-calling agent traces.

Both are far too big per row for a 50M model on CPU (DS2 median is
~16k chars / 11 turns), so every row is clipped on sentence boundaries
before it enters the mix. Sampling is domain-stratified and capped per
source so no single style dominates.

Usage:
    python prepare_data.py                     # defaults, writes data/mix.jsonl
    python prepare_data.py --pt 10 --agent 10 --seed 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as stats
from collections import Counter
from pathlib import Path

import canon

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "mix.jsonl"
SEED_FILE = ROOT / "data" / "seed.jsonl"

DS_PT = "Davizig10jojo/Kimi-K3-And-DeepSeek-V4-Pro-0813-Distillation-in-PT-BR"
DS_AGENT = "zake7749/deepseek-v4-pro-agent-tool-calling-trajectory"

# Clip budgets in chars per profile.
#   cpu   : ~330 tokens/row, fits max_seq 384 on a 12s/step CPU
#   colab : full 1024 ctx, light clipping only
PROFILES = {
    "cpu": dict(think=300, answer=460, prompt=180, sys_pt=160,
                sys_agent=200, msg=240, reason=200, tool=160, budget=850,
                slice_ctx=850),
    "colab": dict(think=900, answer=1600, prompt=320, sys_pt=420,
                  sys_agent=520, msg=900, reason=700, tool=520, budget=2600,
                  slice_ctx=1700),
}
B = dict(PROFILES["cpu"])


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


def _dedupe(rows: list[dict]) -> list[dict]:
    """Drop rows whose rendered content repeats (the agent set has clones)."""
    out, seen = [], set()
    for row in rows:
        digest = hashlib.sha1(" ".join(canon.to_text(row["messages"]).split()).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append(row)
    return out


def _stratified(rows: list[dict], limit: int, key: str, per_group: int, rng: random.Random) -> list[dict]:
    """Round-robin over groups so every domain/style gets representation."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(key) or "na"), []).append(row)
    for items in buckets.values():
        rng.shuffle(items)

    # Round-robin over domains already balances the mix, so scale the cap
    # with the requested limit instead of letting 3-per-domain throttle it.
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


def from_pt(limit: int, rng: random.Random) -> list[dict]:
    """PT-BR reasoning distillations: keep the shape, shrink the length."""
    from datasets import load_dataset

    ds = load_dataset(DS_PT, split="train")
    rows = []
    for row in ds:
        think = canon.sanitize(row.get("thinking"))
        answer = canon.sanitize(row.get("answer"))
        user = canon.sanitize(row.get("user_prompt"))
        system = canon.sanitize(row.get("system_prompt"))

        if len(think) < 120 or len(answer) < 120:
            continue

        msgs = []
        if system and len(system) > 40:
            msgs.append({"role": "system", "content": canon.clip(system, B["sys_pt"])})
        msgs.append({"role": "user", "content": canon.clip(user, B["prompt"])})
        msgs.append(
            {
                "role": "assistant",
                "thinking": canon.clip(think, B["think"]),
                "content": canon.clip(answer, B["answer"]),
            }
        )
        rows.append(
            {
                "messages": msgs,
                "source": "pt-br-reasoning",
                "lang": "pt",
                "domain": row.get("domain") or "na",
            }
        )
    return _stratified(_dedupe(rows), limit, "domain", per_group=3, rng=rng)


def from_agent(limit: int, rng: random.Random, slice_k: int = 1) -> list[dict]:
    """English tool-calling traces, score-filtered.

    slice_k == 1 : keep short trajectories only and trim to one budgeted row.
    slice_k  > 1 : use the full trajectory, then emit turn-level slices so
                   long multi-turn traces become many usable examples.
    """
    from datasets import load_dataset

    ds = load_dataset(DS_AGENT, split="train")
    rows = []
    for row in ds:
        if (row.get("score") or 0) < 0.66:
            continue
        turns = row.get("num_turns") or 0
        if turns < 2 or (slice_k <= 1 and turns > 6):
            continue

        msgs = []
        spent = 0
        limit_msgs = len(row["messages"]) if slice_k > 1 else 8
        for msg in row["messages"][:limit_msgs]:
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

            cost = len(entry.get("content") or "") + len(entry.get("thinking") or "")
            # For single-example mode, stop at the char budget; in slicing
            # mode the per-slice context window enforces it instead.
            if slice_k <= 1 and msgs and spent + cost > B["budget"]:
                break
            msgs.append(entry)
            spent += cost

        # Keep only trajectories that end on an assistant turn.
        while msgs and msgs[-1].get("role") != "assistant":
            msgs.pop()
        if len([m for m in msgs if m.get("role") == "assistant"]) == 0:
            continue

        for variant in slice_turns(msgs, slice_k, B.get("slice_ctx", B["budget"])):
            rows.append(
                {
                    "messages": variant,
                    "source": "agent-tool",
                    "lang": "en",
                    "domain": row.get("domain") or "na",
                }
            )
    return _stratified(_dedupe(rows), limit, "domain", per_group=4, rng=rng)


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
            rows.append({"messages": msgs, "source": "seed", "lang": "en", "domain": "general"})
    rng.shuffle(rows)
    return rows


def token_len(msgs: list[dict], tok) -> int:
    n = 0
    for seg in canon.render(msgs):
        n += len(tok(seg["text"], add_special_tokens=False).input_ids) + (1 if seg["eos"] else 0)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=int, default=14, help="rows from the PT-BR reasoning set")
    ap.add_argument("--agent", type=int, default=12, help="rows from the tool-calling set")
    ap.add_argument("--seed", type=int, default=24, help="rows from data/seed.jsonl")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="cpu",
                    help="clip budget preset: cpu (max_seq 384) or colab (max_seq 1024)")
    ap.add_argument("--pt-repeat", type=int, default=1, help="oversample the small PT set")
    ap.add_argument("--seed-repeat", type=int, default=1, help="oversample the local seed set")
    ap.add_argument("--slice-turns", type=int, default=1,
                    help="split each agent trajectory into up to N turn-level examples")
    ap.add_argument("--max-tokens", type=int, default=384, help="drop rows longer than this")
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    B.update(PROFILES[args.profile])
    if args.profile == "colab" and args.max_tokens == 384:
        args.max_tokens = 1024
    print(f"[*] profile={args.profile} slice_turns={args.slice_turns} max_tokens={args.max_tokens}")

    tok = AutoTokenizer.from_pretrained(ROOT / "model")
    rng = random.Random(args.rng_seed)

    print("[*] sampling PT-BR reasoning ...")
    pt = from_pt(args.pt, rng)
    print(f"    kept {len(pt)}")
    print("[*] sampling agent tool-calling ...")
    agent = from_agent(args.agent, rng, slice_k=args.slice_turns)
    print(f"    kept {len(agent)}")
    print("[*] sampling local seed ...")
    seed = from_seed(rng)[: args.seed]
    print(f"    kept {len(seed)}")

    records, seen, dropped, overlong = [], set(), 0, 0
    for row in pt + agent + seed:
        text = canon.to_text(row["messages"])
        # Hash the whole rendered row: system prompts repeat across rows,
        # so prefix-only hashing would falsely merge distinct examples.
        digest = hashlib.sha1(" ".join(text.split()).encode()).hexdigest()
        if digest in seen:
            dropped += 1
            continue
        n = token_len(row["messages"], tok)
        if n > args.max_tokens:
            overlong += 1
            continue
        seen.add(digest)
        row["tokens"] = n
        records.append(row)

    # Upweighting happens *after* dedupe, so repeats survive as real repeats
    # rather than being collapsed by the hash filter.
    for src, times in (("pt-br-reasoning", args.pt_repeat), ("seed", args.seed_repeat)):
        base = [r for r in records if r["source"] == src]
        for _ in range(max(0, times - 1)):
            records.extend({**r} for r in base)

    rng.shuffle(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lens = [r["tokens"] for r in records]
    by_src = Counter(r["source"] for r in records)
    tok_share = Counter()
    for r in records:
        tok_share[r["source"]] += r["tokens"]
    total_tok = sum(tok_share.values()) or 1
    print(f"\n[+] wrote {len(records)} rows -> {args.out}  (dup {dropped}, overlong {overlong})")
    print(f"    mix      : {dict(by_src)}")
    print(f"    tok share: " + ", ".join(f"{k} {100 * v / total_tok:.0f}%" for k, v in tok_share.most_common()))
    print(f"    tokens   : med {int(stats.median(lens))} / max {max(lens)} / total {sum(lens)}")
    print(f"    domains  : {dict(Counter(r['domain'] for r in records).most_common(8))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
