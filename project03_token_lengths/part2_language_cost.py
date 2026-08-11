"""Part 2 -- what the token-length distribution says about cost, across nine languages.

Project 1 measured that Telugu costs GPT-2 roughly 14x more tokens than English.
Project 2 measured chars-per-token for English documents. This part explains both by
looking at the shape of the distribution rather than its mean.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from corpus import multilingual_corpus
from fingerprint import MAX_LEN, fingerprint

plt.rcParams.update({
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "savefig.dpi": 200,
})

# Representative mid-2026 API input rate, same basis as Project 2. Edit for current pricing.
RATE_IN = 3.00 / 1_000_000  # $ per input token


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    corpus = multilingual_corpus()

    rows, hists = [], {}
    for name, d in corpus.items():
        fp = fingerprint(tokenizer, d["text"])
        hists[name] = fp["hist"]
        rows.append(dict(
            language=name, script=d["script"],
            chars=fp["n_chars"], tokens=fp["n_tokens"],
            chars_per_token=fp["chars_per_token"],
            mean_surface_len=fp["mean_surface_len"],
            pct_len1=100 * fp["hist"][0],
            pct_len_le2=100 * fp["hist"][:2].sum(),
        ))

    df = pd.DataFrame(rows)
    english_cpt = float(df.loc[df.language == "English", "chars_per_token"].iloc[0])
    df["cost_multiplier"] = english_cpt / df["chars_per_token"]
    # Cost of one 10,000-character document at the reference input rate.
    df["usd_per_10k_chars"] = 10_000 / df["chars_per_token"] * RATE_IN
    df = df.sort_values("cost_multiplier").reset_index(drop=True)

    pd.set_option("display.width", 200)
    print("\n  " + df.drop(columns=["chars"]).to_string(index=False,
          float_format=lambda v: f"{v:.3f}").replace("\n", "\n  "))

    latin = df[df.script == "Latin"]
    indic = df[df.script == "Indic"]
    print(f"\n  Latin script : {latin.chars_per_token.min():.2f}-"
          f"{latin.chars_per_token.max():.2f} chars/token, "
          f"tokens of length 1 are {latin.pct_len1.min():.1f}-{latin.pct_len1.max():.1f}% of all tokens")
    print(f"  Indic script : {indic.chars_per_token.min():.2f}-"
          f"{indic.chars_per_token.max():.2f} chars/token, "
          f"tokens of length 1 are {indic.pct_len1.min():.1f}-{indic.pct_len1.max():.1f}% of all tokens")
    print(f"  cost multiplier vs English: Indic {indic.cost_multiplier.min():.1f}x-"
          f"{indic.cost_multiplier.max():.1f}x, Latin up to {latin.cost_multiplier.max():.1f}x")

    # ---- figure 1: the distributions, which is where the separation is visible ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(1, MAX_LEN + 1)
    for name in df.language:
        script = df.loc[df.language == name, "script"].iloc[0]
        style = dict(color="#c44e52", lw=2.2, marker="o") if script == "Indic" else \
                dict(color="#4c72b0", lw=1.2, alpha=0.55, marker=".")
        axes[0].plot(x, 100 * hists[name], label=f"{name} ({script})", **style)
    axes[0].set(xlabel="Token length (byte-alphabet characters)", ylabel="Share of tokens (%)",
                title="Two scripts, two shapes", xticks=range(0, MAX_LEN + 1, 2))
    axes[0].legend(fontsize=7, frameon=False, ncol=2)

    colors = ["#c44e52" if s == "Indic" else "#4c72b0" for s in df.script]
    axes[1].barh(df.language, df.cost_multiplier, color=colors, edgecolor="k")
    axes[1].axvline(1.0, color="k", ls="--", lw=1)
    for i, v in enumerate(df.cost_multiplier):
        axes[1].text(v + 0.15, i, f"{v:.1f}x", va="center", fontsize=8)
    axes[1].set(xlabel="Token cost multiplier vs English", title="What that shape costs")

    plt.tight_layout()
    plt.savefig("part2_language_cost.png")
    plt.close()

    df.to_csv("part2_language_cost.csv", index=False)
    np.save("part2_histograms.npy", np.vstack([hists[n] for n in df.language]))
    print("\n  wrote part2_language_cost.png, part2_language_cost.csv")


if __name__ == "__main__":
    main()
