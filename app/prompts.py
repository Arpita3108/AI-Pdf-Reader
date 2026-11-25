"""Prompt construction. Grounding rules live here."""
from __future__ import annotations

NOT_FOUND = "I couldn't find an answer to that in the uploaded documents."

SYSTEM_PROMPT = f"""You are a careful research assistant. Answer the user's question \
using ONLY the numbered context passages provided.

Rules:
1. Use only facts stated in the passages. Never add outside knowledge or guesses.
2. After every claim, cite the passage number(s) in square brackets, e.g. [1] or [2][3].
3. If the passages do not contain the answer, reply exactly: {NOT_FOUND}
4. If passages conflict, say so and cite both.
5. Be concise and precise."""


def format_context(contexts: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        blocks.append(f"[{i}] (source: {c['doc']}, page {c['page']})\n{c['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, contexts: list[dict]) -> list[dict]:
    user = f"Context passages:\n\n{format_context(contexts)}\n\nQuestion: {question}\n\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
