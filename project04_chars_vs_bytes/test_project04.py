"""Tests. Run: python -m pytest test_project04.py -q"""
from pathlib import Path

import numpy as np
import pytest

from vocab import bytes_per_token, load, vocab_lengths

CACHE = Path(__file__).parent.parent / "project03_token_lengths" / "cache"


@pytest.fixture(scope="module")
def tok():
    return load()


@pytest.fixture(scope="module")
def lengths(tok):
    return vocab_lengths(tok)


def test_utf8_byte_counts_are_what_we_think():
    """The book's warm-up. If this is wrong, everything downstream is."""
    assert len("x".encode("utf-8")) == 1
    assert len("ü".encode("utf-8")) == 2
    assert len("活".encode("utf-8")) == 3
    assert len("🥰".encode("utf-8")) == 4
    assert len("తె".encode("utf-8")) == 6      # two Telugu code points, 3 bytes each


def test_bytes_are_never_fewer_than_characters(lengths):
    n_chars, n_bytes = lengths
    assert (n_bytes >= 1).all()
    assert n_bytes.sum() > n_chars.sum()


@pytest.mark.parametrize("name", ["gutenberg_84_Frankenstein.txt", "wikipedia_te.txt"])
def test_byte_lengths_are_additive_over_real_text(tok, lengths, name):
    """The whole point of the project: byte lengths sum exactly, to the byte."""
    _, n_bytes = lengths
    text = (CACHE / name).read_text(encoding="utf-8")[:50_000]
    ids = tok(text)["input_ids"]
    assert int(n_bytes[ids].sum()) == len(text.encode("utf-8"))


def test_character_lengths_are_NOT_additive(tok, lengths):
    """The trap. A Telugu character spans three tokens, and decoding one token alone
    returns U+FFFD, so the per-token character table over-counts badly. Asserting the
    failure keeps anyone (including me) from 'fixing' it into a wrong identity."""
    n_chars, _ = lengths
    text = (CACHE / "wikipedia_te.txt").read_text(encoding="utf-8")[:50_000]
    ids = tok(text)["input_ids"]
    assert int(n_chars[ids].sum()) > 2 * len(text)


def test_single_telugu_token_does_not_round_trip(tok):
    """Why the table fails: one byte of a 3-byte character decodes to the replacement char."""
    ids = tok("తెలుగు")["input_ids"]
    assert len(ids) > len("తెలుగు")
    assert tok.decode([ids[0]]) == "�"
    assert tok.decode(ids) == "తెలుగు"          # together they are fine


def test_bytes_per_token_matches_a_hand_count(tok, lengths):
    _, n_bytes = lengths
    text = "hello"
    ids = tok(text)["input_ids"]
    assert bytes_per_token(text, tok, n_bytes) == pytest.approx(5 / len(ids))


def test_telugu_costs_several_times_more_than_english(tok):
    en = (CACHE / "gutenberg_84_Frankenstein.txt").read_text(encoding="utf-8")[:50_000]
    te = (CACHE / "wikipedia_te.txt").read_text(encoding="utf-8")[:50_000]
    cpt = lambda t: len(t) / len(tok(t)["input_ids"])
    assert cpt(en) / cpt(te) > 5
