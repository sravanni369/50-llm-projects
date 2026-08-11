"""Part 1 -- the book's figure: character length vs byte length for all 50,257 tokens."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from vocab import load, vocab_lengths

plt.rcParams.update({"axes.spines.right": False, "axes.spines.top": False,
                     "axes.titleweight": "bold", "axes.labelweight": "bold"})


def main() -> None:
    tok = load()
    n_chars, n_bytes = vocab_lengths(tok)

    grid = np.zeros((n_chars.max() + 1, n_bytes.max() + 1))
    for c, b in zip(n_chars, n_bytes):
        grid[c, b] += 1

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    norm = mpl.colors.Normalize(vmin=0, vmax=np.log(grid.max()))
    x, y = np.nonzero(grid)
    for xi, yi in zip(x, y):
        ax[0].plot(xi, yi, "o", markersize=6, markeredgecolor="k", markeredgewidth=0.3,
                   markerfacecolor=plt.cm.magma(norm(np.log(grid[xi, yi]))))
    lim = max(n_chars.max(), n_bytes.max())
    ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="characters = bytes")
    ax[0].set(xlabel="Token length (characters)", ylabel="Token length (bytes)",
              title="Most tokens sit on the line; the rest sit above it")
    ax[0].legend(frameon=False, fontsize=9)
    fig.colorbar(mpl.cm.ScalarMappable(cmap=mpl.cm.magma, norm=norm), ax=ax[0],
                 pad=0.01, label="Log frequency")

    diff = n_bytes - n_chars
    vals, counts = np.unique(diff, return_counts=True)
    ax[1].bar(vals, counts, color="#2a78d6", edgecolor="k", linewidth=0.4)
    ax[1].set_yscale("log")
    ax[1].set(xlabel="Extra bytes beyond characters", ylabel="Tokens (log scale)",
              title=f"{int((diff != 0).sum()):,d} of {tok.vocab_size:,d} tokens "
                    f"cost more bytes than characters")

    plt.tight_layout()
    plt.savefig("chars_vs_bytes.png", dpi=200)
    plt.close()

    ne = diff != 0
    print(f"\n  vocabulary            {tok.vocab_size:,d} tokens")
    print(f"  characters == bytes   {int((~ne).sum()):,d}  ({(~ne).mean():.1%})")
    print(f"  characters != bytes   {int(ne.sum()):,d}  ({ne.mean():.1%})")
    print(f"  worst offender        {int(diff.max())} extra bytes")
    print(f"  longest token         {n_chars.max()} characters, {n_bytes.max()} bytes")
    longest = int(np.argmax(n_bytes))
    print(f"  that token decodes to {tok.decode([longest])!r}")
    print("\n  wrote chars_vs_bytes.png")


if __name__ == "__main__":
    main()
