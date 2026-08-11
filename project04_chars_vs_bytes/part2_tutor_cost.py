"""What an AI tutor costs a Telugu-medium child versus an English-medium one.

The tokenizer bills per token. A Telugu character is 3 UTF-8 bytes and GPT-2 has almost
no Telugu merges, so it bills close to per byte. Same lesson, different price.

Text is the Wikipedia prose cached by project 3 -- a proxy for school reading material,
not actual textbooks. Stated because it bounds how far the number generalises.
"""
from __future__ import annotations

import sys
from pathlib import Path

from vocab import bytes_per_token, load, vocab_lengths

CACHE = Path(__file__).parent.parent / "project03_token_lengths" / "cache"
RATE_IN = 3.00 / 1_000_000        # $ per input token, mid-2026 reference rate
USD_INR = 88.0
LESSON_CHARS = 4_000              # one tutoring exchange, roughly two pages of prose

TEXTS = {
    "English": "gutenberg_84_Frankenstein.txt",
    "Telugu": "wikipedia_te.txt",
    "Hindi": "wikipedia_hi.txt",
    "Tamil": "wikipedia_ta.txt",
}


def main() -> None:
    if not CACHE.exists():
        sys.exit(f"corpus missing: run project 3's corpus.py first ({CACHE})")

    tok = load()
    n_chars, n_bytes = vocab_lengths(tok)

    print(f"\nGPT-2 vocabulary: {tok.vocab_size:,d} tokens")
    ne = int((n_chars != n_bytes).sum())
    print(f"  tokens where characters != bytes: {ne:,d}  ({ne/tok.vocab_size:.1%})")
    print(f"  longest token: {n_chars.max()} characters, {n_bytes.max()} bytes")

    texts = {k: (CACHE / v).read_text(encoding="utf-8")[:200_000] for k, v in TEXTS.items()}

    # --- which length is additive ---
    print("\nIs the vocabulary table additive over a real text?")
    print(f"  {'language':<10}{'sum(bytes)':>12}{'len(utf-8)':>12}{'exact?':>8}"
          f"{'sum(chars)':>12}{'len(text)':>11}{'exact?':>8}")
    for name, text in texts.items():
        ids = tok(text)["input_ids"]
        sb, lb = int(n_bytes[ids].sum()), len(text.encode("utf-8"))
        sc, lc = int(n_chars[ids].sum()), len(text)
        print(f"  {name:<10}{sb:>12,d}{lb:>12,d}{'yes' if sb == lb else 'NO':>8}"
              f"{sc:>12,d}{lc:>11,d}{'yes' if sc == lc else 'NO':>8}")

    # --- the cost, computed from the text itself ---
    print(f"\nCost of one {LESSON_CHARS:,d}-character lesson at ${RATE_IN*1e6:.2f}/M input tokens")
    print(f"  {'language':<10}{'chars/token':>13}{'bytes/token':>13}{'tokens':>9}"
          f"{'INR':>8}{'vs English':>12}")
    base = None
    for name, text in texts.items():
        ids = tok(text)["input_ids"]
        cpt = len(text) / len(ids)                 # measured on the text, not the table
        bpt = bytes_per_token(text, tok, n_bytes)
        n = LESSON_CHARS / cpt
        inr = n * RATE_IN * USD_INR
        base = base or inr
        print(f"  {name:<10}{cpt:>13.3f}{bpt:>13.3f}{n:>9.0f}{inr:>8.2f}{inr/base:>11.1f}x")

    # --- a school year ---
    lessons = 2 * 200
    en = len(texts["English"]) / len(tok(texts["English"])["input_ids"])
    te = len(texts["Telugu"]) / len(tok(texts["Telugu"])["input_ids"])
    en_year = lessons * LESSON_CHARS / en * RATE_IN * USD_INR
    te_year = lessons * LESSON_CHARS / te * RATE_IN * USD_INR
    print("\nA year of tutoring: 2 lessons a day, 200 school days")
    print(f"  English medium  INR {en_year:8.2f}")
    print(f"  Telugu medium   INR {te_year:8.2f}   ({te_year/en_year:.1f}x)")
    print(f"  On the same budget the Telugu-medium child gets "
          f"{en/te:.1f}x fewer lessons.")


if __name__ == "__main__":
    main()
