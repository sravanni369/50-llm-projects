# Project 1 — Three Tokenization Schemes

Own implementation of Project 1 from *50 ML Projects To Understand LLMs* by Mike X Cohen,
based on his MIT-licensed companion code ([mikexcohen/ML4LLM_book](https://github.com/mikexcohen/ML4LLM_book)).
Concept: the same text tokenized three ways — characters, words, GPT-2 byte-level BPE.

## What was built

- `schemes.py` — character, word, and GPT-2 tokenizers behind one small API
- `part1_book_exercise.py` — the book's exercise: one sentence, three schemes, comparison plots
- `part2_telugu_cost.py` — same meaning in English vs Telugu: what does Telugu cost in GPT-2 tokens?
- `part3_jd_vocab.py` — how do data-analyst skill terms fragment under BPE?
- `test_tokenizers.py` — 8 pytest tests (roundtrips, vocab properties, leading-space effect, Telugu cost)

## Results (honest scope)

**Part 1 (book exercise).** "The way you do anything is the way you do everything." =
53 characters (18 unique) / 11 words (8 unique) / 12 GPT-2 tokens (9 unique).
`' Mike'` and `'Mike'` get entirely different GPT-2 token ids — the leading space is part of the token.

**Part 2 (Telugu token cost).** On **3 author-written sentence pairs** (small, illustrative
sample — not a corpus study): 353 Telugu vs 25 English GPT-2 tokens for the same meanings,
a **~14x inflation**. Telugu fertility ≈ 20 tokens/word vs ≈ 1.1 for English, because GPT-2's
BPE has essentially no Telugu merges and each Telugu character is 3 UTF-8 bytes.
Practical reading: GPT-2-tokenizer-based pricing/context budgets punish Indic-language text badly.

**Part 3 (JD skill terms).** On 3 representative (synthetic, not scraped) requirement lines:
`SQL`, `Python`, `Excel`, `dashboard`, `statistics` survive as single GPT-2 tokens;
`Tableau` → 2 pieces, `PyTorch` → 3, `scikit-learn` → 5.

## Run it

```
.venv\Scripts\python.exe part1_book_exercise.py
.venv\Scripts\python.exe part2_telugu_cost.py
.venv\Scripts\python.exe part3_jd_vocab.py
.venv\Scripts\python.exe -m pytest test_tokenizers.py -v
```

Proof of run: [`screenshots/project01_run.png`](screenshots/project01_run.png)
(VS Code, all three parts + 8/8 tests passing).

Dependencies: `numpy matplotlib transformers pytest` (tokenizer only — no torch needed).
