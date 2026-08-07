"""PyTorch cost estimator: predict token count from character count.

The book's project 2 shows characters, words, and tokens scale together
almost perfectly across documents. If that correlation is real, a
one-parameter linear model should predict tokens from raw character
count alone - which means a budgeting tool can estimate an API bill
without shipping a 50k-vocab tokenizer to the user's phone.

This trains y = a*x + b with plain SGD on the per-paragraph pairs
produced by analyze.py and reports honest error numbers.

Run:  python analyze.py && python estimator.py
Output: fit parameters + MAE on stdout, part3_estimator.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).parent
EPOCHS = 400
LR = 0.05


def main():
    pairs = json.loads((HERE / "char_token_pairs.json").read_text())
    x = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    y = torch.tensor([p[1] for p in pairs], dtype=torch.float32)

    # standardize x for stable SGD, recover raw-scale params afterwards
    mu, sigma = x.mean(), x.std()
    xn = (x - mu) / sigma

    a = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([a, b], lr=LR)

    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = torch.mean((a * xn + b - y) ** 2)
        loss.backward()
        opt.step()

    # convert back to tokens = slope*chars + intercept
    slope = (a / sigma).item()
    intercept = (b - a * mu / sigma).item()

    pred = slope * x + intercept
    mae = torch.mean(torch.abs(pred - y)).item()
    mape = torch.mean(torch.abs(pred - y) / y).item() * 100

    print(f"paragraphs: {len(pairs)}")
    print(f"fit: tokens = {slope:.4f} * chars + {intercept:.2f}")
    print(f"  (i.e. about {1/slope:.2f} characters per GPT-2 token)")
    print(f"MAE:  {mae:.1f} tokens per paragraph")
    print(f"MAPE: {mape:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=18, alpha=0.7, label="paragraphs (5 household docs)")
    xs = torch.linspace(0, x.max() * 1.05, 50)
    ax.plot(xs, slope * xs + intercept, "r-",
            label=f"PyTorch fit: tokens = {slope:.3f}*chars + {intercept:.1f}")
    ax.set_xlabel("characters")
    ax.set_ylabel("GPT-2 tokens")
    ax.set_title(f"Token count is predictable from characters alone (MAE {mae:.1f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "part3_estimator.png", dpi=160)
    print("wrote part3_estimator.png")


if __name__ == "__main__":
    main()
