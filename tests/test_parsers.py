"""Parser smoke tests with fake rows: no network, no model download.

    python tests/test_parsers.py        # or: pytest -q tests/

Each case guards a bug that actually bit during the v4 data build:
  * glaive arguments are a single-quoted JSON string -> json.loads fails
  * glaive often omits </functioncall> -> regex needs brace scanning
  * gsm8k `<<12*2=24>>24` markers must be deleted, not substituted
  * OpenThoughts template markers must be split before sanitize
  * long reasoning traces must be dropped, never truncated
"""

from __future__ import annotations

import pathlib
import random
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import canon  # noqa: E402
import prepare_data as pd  # noqa: E402

GLAIVE_SYS = """SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "get_news_headlines",
    "description": "Get the latest news headlines"
}
"""

GLAIVE_CHAT = """USER: Can you tell me the latest news headlines for the United States?


ASSISTANT: <functioncall> {"name": "get_news_headlines", "arguments": '{"country": "United States"}'} <|endoftext|>


FUNCTION RESPONSE: {"headlines": ["Biden announces new vaccine mandates", "Apple unveils new iPhone"]}


ASSISTANT: Here are the latest headlines:
1. Biden announces new vaccine mandates
2. Apple unveils new iPhone <|endoftext|>
"""


def fake_dataset(rows):
    """Patch datasets.load_dataset so source functions see `rows`."""
    class _DS:
        def __iter__(self):
            return iter(rows)

        def __len__(self):
            return len(rows)

    sys.modules["datasets"] = types.SimpleNamespace(load_dataset=lambda *a, **k: _DS())


def roles(msgs):
    return [m.get("role") for m in msgs]


def test_glaive_keeps_tool_calls_and_results():
    fake_dataset([{"system": GLAIVE_SYS, "chat": GLAIVE_CHAT}])
    rows = pd.from_agent_glaive(10, random.Random(0), slice_k=3)
    assert rows, "glaive produced no rows"
    flat = [m for r in rows for m in r["messages"]]
    calls = [m for m in flat if m.get("tool_calls")]
    assert calls, "tool_calls lost (single-quoted arguments or missing </functioncall>)"
    assert calls[0]["tool_calls"][0]["function"]["name"] == "get_news_headlines"
    assert any(m.get("role") == "tool" for m in flat), "tool results lost"
    # the call-turn slice must not leak the <functioncall> tag into the text
    assert "<functioncall>" not in canon.to_text(flat)


def test_gsm8k_markers_deleted_not_substituted():
    fake_dataset([{
        "question": "Ann has 3 apples and buys 4 more. How many?",
        "answer": ("Ann starts with three apples in her basket.\n"
                   "She then buys four more apples from the market.\n"
                   "So the total is 3 + 4 = <<3+4=7>>7 apples.\n#### 7"),
    }])
    rows = pd.from_math_gsm8k(5, random.Random(0))
    assert rows, "gsm8k produced no rows"
    text = canon.to_text(rows[0]["messages"])
    assert "<<" not in text, "calculator markers survived"
    assert "= 77" not in text, "marker value was substituted, duplicating the number"
    assert "= 7 apples" in text
    assert rows[0]["messages"][-1]["content"] == "7"


def test_mathinstruct_splits_reasoning_from_answer():
    fake_dataset([{
        "source": "data/CoT/gsm_rft.json",
        "instruction": "A pen costs 2 dollars. What do 5 pens cost?",
        "output": ("Each pen costs two dollars, so five pens cost five times two, "
                   "which we compute as 5 * 2 = 10. The answer is 10"),
    }])
    rows = pd.from_math_instruct(5, random.Random(0))
    assert rows, "MathInstruct produced no rows"
    last = rows[0]["messages"][-1]
    assert last["content"] == "10"
    assert "five times two" in last["thinking"]


def test_code_magicoder_splits_prose_comment_into_thinking():
    code = ("# Approach: sort the array, then walk it once keeping a running sum.\n"
            "# The answer is the maximum of that sum over all prefixes.\n"
            "def solve(a):\n"
            "    best = total = 0\n"
            "    for x in sorted(a):\n"
            "        total += x\n"
            "        best = max(best, total)\n"
            "    return best\n")
    fake_dataset([{"problem": "Given an array, return the best prefix sum.", "solution": code, "lang": "python"}])
    rows = pd.from_code_magicoder(5, random.Random(0))
    assert rows, "magicoder produced no rows"
    last = rows[0]["messages"][-1]
    assert "sort the array" in last["thinking"], "prose comment was not split into thinking"
    assert "def solve" in last["content"], "code lost"
    assert "# Approach" not in last["content"], "comment leaked into the answer"
    assert rows[0]["domain"] == "python"


def test_code_magicoder_drops_overlong_rows_instead_of_truncating():
    fake_dataset([{"problem": "p" * 5000, "solution": "def f(): pass", "lang": "python"},
                  {"problem": "q", "solution": "x = 1\n" * 5000, "lang": "python"}])
    assert not pd.from_code_magicoder(5, random.Random(0))


def test_agent_deepseek_takes_longest_prefix_per_id():
    """Rows sharing an id are prefixes; keep the whole conversation, not the first turn."""
    base = {"id": "1001", "domain": "bank", "language": "en", "score": 1.0, "num_turns": 3}
    s = {"role": "system", "content": "You are a bank agent."}
    u1 = {"role": "user", "content": "What is my balance?"}
    a1 = {"role": "assistant", "content": "Let me check.", "reasoning": "Need the account id.",
          "tool_calls": [{"function": {"name": "get_balance", "arguments": "{}"}}]}
    t1 = {"role": "tool", "content": "{\"balance\": 42}", "name": "get_balance"}
    u2 = {"role": "user", "content": "Thanks, and the last transaction?"}
    a2 = {"role": "assistant", "content": "It was 12 dollars.", "reasoning": "Read the tool result."}

    prefixes = [
        {**base, "messages": [s, u1, a1]},
        {**base, "messages": [s, u1, a1, t1, a2]},
        {**base, "messages": [s, u1, a1, t1, a2, u2, {**a2, "content": "Anything else?"}]},
    ]
    fake_dataset(prefixes)
    rows = pd.from_agent_deepseek(50, random.Random(0), slice_k=3)
    assert rows, "agent-deepseek produced no rows"
    turns = max(sum(1 for m in r["messages"] if m["role"] == "assistant") for r in rows)
    assert turns >= 2, "only the one-turn prefix survived: multi-turn slice lost"
    assert any(m.get("role") == "tool" for r in rows for m in r["messages"]), "tool results lost"
    assert any(m.get("thinking") for r in rows for m in r["messages"]), "per-turn reasoning lost"


def test_language_gate():
    assert canon.is_english(
        "The model predicts the next token given the context window and the prompt.")
    assert not canon.is_english(
        "Não consigo atualizar o endereço de entrega porque o sistema está fora do ar e você "
        "precisa ligar para o atendimento ao cliente amanhã de manhã, por favor.")
    assert not canon.is_english(
        "Je ne peux pas mettre à jour l'adresse de livraison parce que le système est hors "
        "service et vous devez appeler le service client demain matin, s'il vous plaît.")
    assert canon.is_english("3 + 4 = 7 and the answer is 7")


def test_supervised_spans_only():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    segs = canon.render(msgs)
    assert [s["train"] for s in segs] == [False, False, True]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")
    print("\nALL TESTS PASSED" if not failures else f"\n{failures} FAILED")
    sys.exit(1 if failures else 0)
