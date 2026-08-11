"""Tests for project 3. Run: python -m pytest test_project03.py -q

Everything that needs the corpus reads it from `cache/`, so the suite is offline after
the first `python corpus.py`.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from corpus import multilingual_corpus, strip_gutenberg_boilerplate
from fingerprint import MAX_LEN, fingerprint, histogram, token_surface_lengths, windows
from model import CostRegressor, LanguageClassifier, train
from part1_book_exercise import PASSAGE, frequency_table, token_lengths


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def corpus():
    return multilingual_corpus()


@pytest.fixture(scope="module")
def hists(tok, corpus):
    return {n: fingerprint(tok, d["text"])["hist"] for n, d in corpus.items()}


# ---------------------------------------------------------------- tokenizing

def test_tokenizer_round_trips(tok):
    ids = tok(PASSAGE)["input_ids"]
    assert tok.decode(ids) == PASSAGE


def test_token_lengths_match_token_count(tok):
    lens = token_lengths(tok, PASSAGE)
    assert len(lens) == len(tok(PASSAGE)["input_ids"])
    assert all(l >= 1 for l in lens)


def test_frequency_table_is_sorted_and_totals(tok):
    lens = token_lengths(tok, PASSAGE)
    counts = frequency_table(lens)
    assert list(counts.index) == sorted(counts.index)
    assert counts.sum() == len(lens)


# ---------------------------------------------------------------- fingerprint

def test_histogram_is_a_probability_vector():
    h = histogram(np.array([1, 1, 2, 3, 3, 3]))
    assert len(h) == MAX_LEN
    assert h.sum() == pytest.approx(1.0)
    assert h[0] == pytest.approx(2 / 6) and h[2] == pytest.approx(3 / 6)


def test_histogram_lumps_the_long_tail():
    """A 999-character token must land in the last bin, not off the end of the array."""
    h = histogram(np.array([1, 999]))
    assert h[-1] == pytest.approx(0.5)


def test_histogram_is_length_invariant(tok, corpus):
    """Doubling the text must not move the fingerprint -- that is why it is normalised."""
    text = corpus["English"]["text"][:20_000]
    a = fingerprint(tok, text)["hist"]
    b = fingerprint(tok, text + text)["hist"]
    assert np.abs(a - b).sum() < 0.01


def test_windows_do_not_overlap():
    w = windows("abcdefghij", size=3, stride=3)
    assert w == ["abc", "def", "ghi"]   # the ragged tail is dropped, not padded


# ---------------------------------------------------------------- corpus hygiene

def test_gutenberg_boilerplate_is_stripped():
    raw = ("header junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nreal text\n"
           "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicence junk")
    assert strip_gutenberg_boilerplate(raw) == "real text"


def test_corpus_is_roughly_balanced_within_script(corpus):
    """Indic samples must be big enough to measure; the first run gave 216 chars."""
    for name, d in corpus.items():
        if d["script"] == "Indic":
            assert len(d["text"]) > 100_000, f"{name} sample is too small"


# ---------------------------------------------------------------- the findings

def test_indic_costs_more_tokens_than_english(tok, corpus):
    en = fingerprint(tok, corpus["English"]["text"])["chars_per_token"]
    te = fingerprint(tok, corpus["Telugu"]["text"])["chars_per_token"]
    assert en / te > 5, "Telugu should cost several times more tokens per character"


def test_indic_is_dominated_by_single_byte_tokens(hists, corpus):
    for name, d in corpus.items():
        share = hists[name][0]
        if d["script"] == "Indic":
            assert share > 0.45, f"{name}: expected mostly length-1 tokens, got {share:.2f}"
        else:
            assert share < 0.35, f"{name}: unexpectedly many length-1 tokens ({share:.2f})"


def test_telugu_and_tamil_are_indistinguishable_by_histogram(hists):
    """The documented failure: GPT-2 has no merges for either script, so both collapse
    to pure byte tokens and the two fingerprints coincide. This is why the classifier
    scores 0% on Telugu, and it is a property of the tokenizer, not a training bug."""
    te_ta = np.abs(hists["Telugu"] - hists["Tamil"]).sum()
    others = [np.abs(hists[a] - hists[b]).sum()
              for a, b in itertools.combinations(sorted(hists), 2)
              if {a, b} != {"Telugu", "Tamil"}]
    assert te_ta < 0.01
    assert te_ta < min(others) / 10


# ---------------------------------------------------------------- models

def test_classifier_output_shape():
    m = LanguageClassifier(n_classes=9)
    assert m(torch.rand(7, MAX_LEN)).shape == (7, 9)


def test_regressor_is_positive():
    m = CostRegressor()
    out = m(torch.rand(16, MAX_LEN))
    assert out.shape == (16,)
    assert (out > 0).all(), "chars-per-token must be positive"


def test_training_reduces_loss():
    torch.manual_seed(0)
    X = torch.rand(64, MAX_LEN)
    y = (X[:, 0] > 0.5).long()
    m = LanguageClassifier(n_classes=2)
    losses = train(m, X, y, loss_fn=nn.CrossEntropyLoss(), epochs=200)
    assert losses[-1] < losses[0]


def test_training_is_reproducible():
    """`train(seed=)` fixes dropout and update order, but NOT the weight initialisation,
    which happens when the model is constructed. The caller has to seed before building
    the model -- forgetting that is what made two 'identical' runs diverge at epoch 0."""
    X, y = torch.rand(32, MAX_LEN), torch.randint(0, 3, (32,))

    def run():
        torch.manual_seed(1)
        return train(LanguageClassifier(3), X, y, loss_fn=nn.CrossEntropyLoss(),
                     epochs=50, seed=1)

    assert run() == run()
