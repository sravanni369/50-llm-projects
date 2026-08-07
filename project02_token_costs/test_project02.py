"""Sanity checks for project 02. Run: python -m pytest -q"""

import json
from pathlib import Path

from analyze import count_doc, monthly_costs

HERE = Path(__file__).parent


class FakeTokenizer:
    """Deterministic stand-in: one token per whitespace-split word."""
    def encode(self, text):
        return [hash(w) % 1000 for w in text.split()]


def test_count_doc_basic():
    r = count_doc("the cat sat on the mat", FakeTokenizer())
    assert r["chars"] == 22
    assert r["words"] == 6
    assert r["tokens"] == 6
    assert r["uniq_words"] == 5  # "the" repeats


def test_count_doc_empty_uniques():
    r = count_doc("aaa", FakeTokenizer())
    assert r["uniq_chars"] == 1


def test_monthly_costs_chunked_never_more_expensive():
    for doc_tokens in (100, 800, 5000, 25000):
        full, chunked = monthly_costs(doc_tokens)
        assert chunked <= full


def test_monthly_costs_small_doc_equal():
    # doc smaller than one chunk: both strategies send the same context
    full, chunked = monthly_costs(500)
    assert abs(full - chunked) < 1e-9


def test_docs_exist_and_sized():
    docs = list((HERE / "docs").glob("*.txt"))
    assert len(docs) == 5
    sizes = {p.stem: len(p.read_text(encoding="utf-8")) for p in docs}
    assert sizes["benefits_summary"] > 50_000  # handbook-sized
    assert sizes["utility_bill"] < 5_000


def test_pairs_file_valid():
    pairs = json.loads((HERE / "char_token_pairs.json").read_text())
    assert len(pairs) > 100
    assert all(c > 0 and t > 0 for c, t in pairs)
