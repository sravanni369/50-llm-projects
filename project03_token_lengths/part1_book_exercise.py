"""Part 1 -- the book's exercise: pandas frequency tables of GPT-2 token lengths.

Two figures:
  1. Token-length distribution for a single passage.
  2. Normalised token-length distributions for ten Project Gutenberg books, log-x.

The point of figure 2 is that ten books written across two centuries in wildly
different registers produce nearly the same curve, because the tokenizer's vocabulary
-- not the author -- decides where words break.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never try to open a window

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

from corpus import BOOK_URLS, gutenberg

plt.rcParams.update({
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "savefig.dpi": 200,
})

# Same passage the author uses (Wikipedia, CC BY-SA), so figure 1 is comparable.
PASSAGE = (
    "Pulcinella is a 21-section ballet by Igor Stravinsky with arias for soprano, tenor "
    "and bass vocal soloists, and two sung trios. It is based on the 18th-century play "
    "Quatre Polichinelles semblables, or Four similar Pulcinellas, revolving around a "
    "stock character from commedia dell'arte. The work premiered at the Paris Opera on "
    "15 May 1920 under the baton of Ernest Ansermet. The central dancer, Leonide Massine, "
    "created both the libretto and the choreography, while Pablo Picasso designed the "
    "costumes and sets. The ballet was commissioned by Sergei Diaghilev, impresario of "
    "the Ballets Russes. A complete performance takes 35-40 minutes. Stravinsky revised "
    "the score in 1965."
)


def token_lengths(tokenizer, text: str) -> list[int]:
    """Length in characters of every token the tokenizer emits for `text`.

    `convert_ids_to_tokens` returns GPT-2's byte-level surface forms, in which a leading
    space is the sentinel character 'G-with-dot' rather than a real space. That sentinel
    is a genuine part of the token -- 'the' preceded by a space is a different token from
    'the' at the start of a line -- so it is counted, exactly as the book does.
    """
    ids = tokenizer(text)["input_ids"]
    return [len(t) for t in tokenizer.convert_ids_to_tokens(ids)]


def frequency_table(lengths: list[int]) -> pd.Series:
    """Counts per token length, ascending by length (the pandas step of the exercise)."""
    return pd.DataFrame({"length": lengths})["length"].value_counts().sort_index()


def part1_passage(tokenizer) -> pd.Series:
    ids = tokenizer(PASSAGE)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"  passage: {len(PASSAGE):,d} chars -> {len(tokens):,d} tokens "
          f"({len(set(tokens)):,d} unique)")

    # Round-trip check: decoding the ids must give the original string back.
    assert tokenizer.decode(ids) == PASSAGE, "tokenizer did not round-trip"
    print("  round-trip: decode(encode(text)) == text  [ok]")

    counts = frequency_table(token_lengths(tokenizer, PASSAGE))
    print(f"  token lengths run {counts.index.min()}-{counts.index.max()} chars; "
          f"most common is {counts.idxmax()} ({counts.max()} tokens)")

    plt.figure(figsize=(10, 4))
    sns.barplot(x=counts.index, y=counts.values, edgecolor="k",
                hue=counts.values, palette="plasma", legend=False)
    plt.gca().set(xlabel="Token length (characters)", ylabel="Count",
                  title="GPT-2 token lengths in one paragraph")
    plt.tight_layout()
    plt.savefig("part1_passage.png")
    plt.close()
    return counts


def part1_books(tokenizer) -> pd.DataFrame:
    plt.figure(figsize=(10, 4.5))
    markers = "soh^vP*DXp"
    rows = []

    for i, (code, title) in enumerate(BOOK_URLS):
        text = gutenberg(code, title)
        counts = frequency_table(token_lengths(tokenizer, text))
        norm = counts / counts.sum()  # normalise: books differ hugely in length

        sns.scatterplot(x=norm.index, y=norm.values, s=80, alpha=0.5,
                        marker=markers[i], edgecolor="k", label=title)
        rows.append(dict(book=title, chars=len(text), tokens=int(counts.sum()),
                         mean_token_len=round(float((counts.index * counts).sum() / counts.sum()), 3),
                         chars_per_token=round(len(text) / counts.sum(), 3)))
        print(f"  {title:<17} {len(text):>9,d} chars  {counts.sum():>8,d} tokens  "
              f"{len(text)/counts.sum():.2f} chars/token")

    plt.gca().set(xlabel="Token length (characters)", ylabel="Frequency (norm.)",
                  xscale="log", title="Ten books, one tokenizer, one curve")
    plt.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig("part1_books.png")
    plt.close()

    df = pd.DataFrame(rows)
    spread = df["chars_per_token"].max() - df["chars_per_token"].min()
    print(f"\n  chars/token across ten books: {df['chars_per_token'].min():.2f}-"
          f"{df['chars_per_token'].max():.2f}  (spread {spread:.2f})")
    return df


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("\nPart 1a: one passage")
    part1_passage(tokenizer)

    print("\nPart 1b: ten Gutenberg books")
    df = part1_books(tokenizer)
    df.to_csv("part1_books.csv", index=False)
    print("\n  wrote part1_passage.png, part1_books.png, part1_books.csv")


if __name__ == "__main__":
    main()
