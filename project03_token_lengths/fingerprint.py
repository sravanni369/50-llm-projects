"""The token-length fingerprint: a fixed-width, language-agnostic summary of a text.

Everything downstream (Part 2's tables, Part 3's PyTorch models) consumes this one
function, so the feature definition lives in exactly one place.

A note on what "token length" means, because for non-Latin scripts it is not obvious.
GPT-2 is a *byte-level* BPE tokenizer: it never sees characters, only UTF-8 bytes,
which it maps into a printable surrogate alphabet. For English, one surrogate is one
character, so a token of surface length 4 really is four letters. For Telugu, where a
single character occupies three UTF-8 bytes, a token of surface length 3 may be *one*
character. So the histogram measures bytes-per-token, and the chars-per-token figure is
reported separately against real user-visible characters. The gap between those two
numbers is the whole story of why Indic text costs more.
"""

from __future__ import annotations

import numpy as np

MAX_LEN = 20  # lengths 1..19 get their own bin; >=20 is lumped into the last one.


def token_surface_lengths(tokenizer, text: str) -> np.ndarray:
    ids = tokenizer(text)["input_ids"]
    return np.array([len(t) for t in tokenizer.convert_ids_to_tokens(ids)])


def histogram(lengths: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """Normalised frequency of each token length -> a probability vector of size max_len.

    Index i holds the share of tokens of length i+1; the final bin absorbs the tail.
    Normalising is what makes a 600-word news item comparable to a 600-page novel.
    """
    clipped = np.clip(lengths, 1, max_len)
    counts = np.bincount(clipped, minlength=max_len + 1)[1:max_len + 1]
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def fingerprint(tokenizer, text: str) -> dict:
    """Histogram plus the two scalars that matter for cost."""
    lengths = token_surface_lengths(tokenizer, text)
    n_tokens = len(lengths)
    return dict(
        hist=histogram(lengths),
        n_tokens=n_tokens,
        n_chars=len(text),
        chars_per_token=len(text) / n_tokens if n_tokens else 0.0,
        mean_surface_len=float(lengths.mean()) if n_tokens else 0.0,
    )


def windows(text: str, size: int = 4000, stride: int = 4000) -> list[str]:
    """Split a long text into fixed-size character windows.

    Each window becomes one training example, which is what turns nine documents into
    a few hundred samples. Windows do not overlap, so no character appears in both the
    training and test sets.
    """
    return [text[i:i + size] for i in range(0, len(text) - size + 1, stride)]
