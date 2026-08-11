"""Part 3 -- can a 20-number histogram alone identify the language and predict the bill?

Design choices that decide whether the number at the end means anything:

* Balanced classes. Spanish has 14x more text than Telugu; without a per-language cap
  the classifier could score well by learning "guess Spanish".
* Contiguous, not random, train/test split. Windows are taken in document order and the
  last 30% are held out, so test text comes from a different part of the document than
  anything trained on. A random split would put page 4 in train and page 5 in test.
* A separate unseen-document check on nine English books the model never saw at all.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from corpus import BOOK_URLS, gutenberg, multilingual_corpus
from fingerprint import fingerprint, windows
from model import CostRegressor, LanguageClassifier, train

plt.rcParams.update({
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.titleweight": "bold", "axes.labelweight": "bold", "savefig.dpi": 200,
})

WINDOW = 2000     # characters per sample
TEST_FRAC = 0.30
SEED = 0


def build_dataset(tokenizer, corpus: dict):
    """One row per window: histogram features, language label, true chars/token."""
    per_lang = {n: windows(d["text"], WINDOW, WINDOW) for n, d in corpus.items()}
    cap = min(len(w) for w in per_lang.values())   # balance the classes
    print(f"  {cap} windows of {WINDOW:,d} chars per language "
          f"({cap * len(corpus)} samples total)")

    langs = sorted(corpus)
    Xtr, ytr, ctr, Xte, yte, cte = [], [], [], [], [], []
    for li, name in enumerate(langs):
        chunk = per_lang[name][:cap]
        split = int(cap * (1 - TEST_FRAC))
        for i, w in enumerate(chunk):
            fp = fingerprint(tokenizer, w)
            if i < split:
                Xtr.append(fp["hist"]); ytr.append(li); ctr.append(fp["chars_per_token"])
            else:
                Xte.append(fp["hist"]); yte.append(li); cte.append(fp["chars_per_token"])

    t = lambda a, dt: torch.tensor(np.array(a), dtype=dt)
    return (langs,
            t(Xtr, torch.float32), t(ytr, torch.long), t(ctr, torch.float32),
            t(Xte, torch.float32), t(yte, torch.long), t(cte, torch.float32))


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    corpus = multilingual_corpus()

    print("\nBuilding dataset")
    langs, Xtr, ytr, ctr, Xte, yte, cte = build_dataset(tokenizer, corpus)
    print(f"  train {len(Xtr)}  test {len(Xte)}  features {Xtr.shape[1]}")

    # ---------------- classifier ----------------
    print("\nTask 1: which language?")
    torch.manual_seed(SEED)   # seeds the weight init; train(seed=) only covers dropout
    clf = LanguageClassifier(n_classes=len(langs))
    closs = train(clf, Xtr, ytr, loss_fn=nn.CrossEntropyLoss(), epochs=600, lr=0.01, seed=SEED)

    clf.eval()
    with torch.no_grad():
        pred = clf(Xte).argmax(1)
        train_acc = (clf(Xtr).argmax(1) == ytr).float().mean().item()
    acc = (pred == yte).float().mean().item()
    baseline = 1 / len(langs)
    print(f"  train accuracy {train_acc:6.1%}")
    print(f"  test  accuracy {acc:6.1%}   (chance = {baseline:.1%})")

    cm = np.zeros((len(langs), len(langs)), int)
    for t_, p_ in zip(yte.tolist(), pred.tolist()):
        cm[t_, p_] += 1
    per_class = cm.diagonal() / cm.sum(1)
    for name, a in sorted(zip(langs, per_class), key=lambda kv: kv[1]):
        print(f"    {name:<12} {a:5.0%}")

    # script-level accuracy: the question that actually matters for cost
    script_of = {n: corpus[n]["script"] for n in langs}
    script_ok = sum(script_of[langs[t_]] == script_of[langs[p_]]
                    for t_, p_ in zip(yte.tolist(), pred.tolist()))
    print(f"  script (Latin vs Indic) accuracy: {script_ok/len(yte):.1%}")

    # ---------------- regressor ----------------
    print("\nTask 2: how many characters per token?")
    torch.manual_seed(SEED)
    reg = CostRegressor()
    rloss = train(reg, Xtr, ctr, loss_fn=nn.MSELoss(), epochs=1500, lr=0.01, seed=SEED)
    reg.eval()
    with torch.no_grad():
        chat = reg(Xte)
    mae = float((chat - cte).abs().mean())
    mape = float(((chat - cte).abs() / cte).mean() * 100)
    naive = float((cte - ctr.mean()).abs().mean())   # predict the training mean
    print(f"  MAE {mae:.3f} chars/token  ({mape:.1f}% MAPE)")
    print(f"  predict-the-mean baseline MAE {naive:.3f}  ->  {naive/mae:.1f}x better")

    # ---------------- unseen documents ----------------
    print("\nUnseen-document check: nine English books the model never saw")
    hits = []
    for code, title in BOOK_URLS[1:]:
        fp = fingerprint(tokenizer, gutenberg(code, title))
        with torch.no_grad():
            x = torch.tensor(fp["hist"], dtype=torch.float32).unsqueeze(0)
            guess = langs[int(clf(x).argmax(1))]
            cpt = float(reg(x))
        ok = guess == "English"
        hits.append(ok)
        print(f"    {title:<17} -> {guess:<10} {'ok' if ok else 'MISS':<5} "
              f"chars/token predicted {cpt:.2f}, actual {fp['chars_per_token']:.2f}")
    print(f"  {sum(hits)}/{len(hits)} identified as English")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(closs, color="#4c72b0", label="classifier (cross-entropy)")
    ax2 = axes[0].twinx()
    ax2.plot(rloss, color="#c44e52", label="regressor (MSE)")
    ax2.set_ylabel("MSE", color="#c44e52")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Training")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")

    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set(xticks=range(len(langs)), yticks=range(len(langs)),
                xlabel="Predicted", ylabel="True",
                title=f"Confusion matrix ({acc:.0%} accurate)")
    axes[1].set_xticklabels(langs, rotation=90, fontsize=7)
    axes[1].set_yticklabels(langs, fontsize=7)
    for i in range(len(langs)):
        for j in range(len(langs)):
            if cm[i, j]:
                axes[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                             color="white" if cm[i, j] > cm.max() / 2 else "black")

    axes[2].scatter(cte, chat, s=28, alpha=0.6, edgecolor="k", linewidth=0.4, color="#55a868")
    lo, hi = float(cte.min()) * 0.9, float(cte.max()) * 1.05
    axes[2].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[2].set(xlabel="Actual chars/token", ylabel="Predicted",
                title=f"Cost prediction (MAE {mae:.3f})")

    plt.tight_layout()
    plt.savefig("part3_predict.png")
    plt.close()

    pd.DataFrame(cm, index=langs, columns=langs).to_csv("part3_confusion.csv")
    print("\n  wrote part3_predict.png, part3_confusion.csv")


if __name__ == "__main__":
    main()
