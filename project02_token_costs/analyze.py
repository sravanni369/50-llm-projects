"""Project 02 - Book lengths in characters, words, and tokens,
applied to household paperwork and LLM API costs.

Book concept (ML4LLM project 2): count characters, words, and GPT-2 BPE
tokens across a document set; compare totals and see how tightly the
three scale together.

Real-world application: a middle-income household pasting documents into
a pay-per-token AI API pays by the token. Sending the whole document
with every question is the expensive default. Retrieving only the
relevant chunk per question (the core RAG idea) sends far fewer tokens
for the same answer. This script measures both, in dollars.

Rates are REPRESENTATIVE mid-tier API list prices (USD per million
tokens) as of mid-2026; edit RATE_IN / RATE_OUT to current pricing.

Run:  python make_docs.py && python analyze.py
Outputs: counts + cost tables on stdout, part1_counts.png, part2_costs.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

RATE_IN = 3.00    # $ per 1M input tokens (representative)
RATE_OUT = 15.00  # $ per 1M output tokens (representative)
QUESTIONS_PER_MONTH = 12
ANSWER_TOKENS = 300
QUESTION_TOKENS = 40
CHUNK_TOKENS = 800  # retrieved context per question in the chunked setup

HERE = Path(__file__).parent


def count_doc(text, tokenizer):
    """Return dict of char/word/token totals and uniques for one text."""
    tokens = tokenizer.encode(text)
    words = text.split()
    return {
        "chars": len(text), "words": len(words), "tokens": len(tokens),
        "uniq_chars": len(set(text)), "uniq_words": len(set(words)),
        "uniq_tokens": len(set(tokens)),
    }


def monthly_costs(doc_tokens):
    """Dollar cost of a month of Q&A about one document, three ways."""
    out_cost = QUESTIONS_PER_MONTH * ANSWER_TOKENS * RATE_OUT / 1e6
    full = QUESTIONS_PER_MONTH * (doc_tokens + QUESTION_TOKENS) * RATE_IN / 1e6 + out_cost
    chunk_ctx = min(CHUNK_TOKENS, doc_tokens)
    chunked = QUESTIONS_PER_MONTH * (chunk_ctx + QUESTION_TOKENS) * RATE_IN / 1e6 + out_cost
    return full, chunked


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 10**9  # we only count tokens, no model context limit applies
    docs = sorted((HERE / "docs").glob("*.txt"))
    if not docs:
        raise SystemExit("run make_docs.py first")

    rows = {}
    for p in docs:
        rows[p.stem] = count_doc(p.read_text(encoding="utf-8"), tokenizer)

    print(f"{'document':<26} {'chars':>7} {'words':>7} {'tokens':>7} {'uniq_tok':>8}")
    for name, r in rows.items():
        print(f"{name:<26} {r['chars']:>7,} {r['words']:>7,} {r['tokens']:>7,} {r['uniq_tokens']:>8,}")

    # ---- part 1 chart: totals per document ----
    names = list(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(names))
    w = 0.27
    ax.bar([i - w for i in x], [rows[n]["chars"] for n in names], w, label="characters")
    ax.bar(list(x), [rows[n]["words"] for n in names], w, label="words")
    ax.bar([i + w for i in x], [rows[n]["tokens"] for n in names], w, label="GPT-2 tokens")
    ax.set_yscale("log")
    ax.set_xticks(list(x), [n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("count (log scale)")
    ax.set_title("Household documents: characters vs words vs tokens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "part1_counts.png", dpi=160)

    # ---- part 2: what the tokens cost ----
    print(f"\nMonthly cost of {QUESTIONS_PER_MONTH} questions per document "
          f"(rates: ${RATE_IN}/M in, ${RATE_OUT}/M out):")
    print(f"{'document':<26} {'full-doc':>9} {'chunked':>9} {'saved':>7}")
    fulls, chunks = [], []
    for name, r in rows.items():
        full, chunked = monthly_costs(r["tokens"])
        fulls.append(full)
        chunks.append(chunked)
        pct = 100 * (full - chunked) / full
        print(f"{name:<26} ${full:>8.2f} ${chunked:>8.2f} {pct:>6.1f}%")

    total_full, total_chunk = sum(fulls), sum(chunks)
    print(f"{'ALL FIVE DOCUMENTS':<26} ${total_full:>8.2f} ${total_chunk:>8.2f} "
          f"{100*(total_full-total_chunk)/total_full:>6.1f}%")
    print(f"{'PER YEAR':<26} ${12*total_full:>8.2f} ${12*total_chunk:>8.2f} "
          f"  saves ${12*(total_full-total_chunk):.2f}/yr")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - 0.2 for i in x], fulls, 0.4, label="paste full document each question")
    ax.bar([i + 0.2 for i in x], chunks, 0.4, label="retrieve one relevant chunk (RAG)")
    ax.set_xticks(list(x), [n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel(f"$ per month ({QUESTIONS_PER_MONTH} questions)")
    ax.set_title("Same questions, same answers, different token bills")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "part2_costs.png", dpi=160)

    # per-paragraph (chars, tokens) pairs for the estimator
    pairs = []
    for p in docs:
        for para in p.read_text(encoding="utf-8").split("\n\n"):
            if para.strip():
                pairs.append([len(para), len(tokenizer.encode(para))])
    (HERE / "char_token_pairs.json").write_text(json.dumps(pairs))
    print(f"\nwrote char_token_pairs.json ({len(pairs)} paragraphs) for estimator.py")


if __name__ == "__main__":
    main()
