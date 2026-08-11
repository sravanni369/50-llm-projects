"""Fetch and cache the text this project measures.

Two sources, both chosen so the project can be re-run by anyone:

* Project Gutenberg for Latin-script languages -- public domain, stable URLs.
* Wikipedia extracts for Indic scripts, because Gutenberg has almost no Telugu.
  Only plain-text extracts are cached locally; nothing is redistributed in the repo
  (see .gitignore), and only the derived statistics are committed.

Every download is cached under `cache/`, so the analysis scripts run offline after the
first pass. Nothing here reproduces book text.
"""

from __future__ import annotations

import json
import time
import re
from pathlib import Path

import requests

CACHE = Path(__file__).parent / "cache"
UA = {"User-Agent": "ml4llm-project3/1.0 (educational; token-length study)"}

# --- Latin script: Project Gutenberg -----------------------------------------
# The ten books the author uses, kept as-is so Part 1 is a faithful reproduction.
BOOK_URLS = [
    ("84", "Frankenstein"),
    ("64317", "GreatGatsby"),
    ("11", "AliceWonderland"),
    ("1513", "RomeoJuliet"),
    ("76", "HuckFinn"),
    ("219", "HeartDarkness"),
    ("2591", "GrimmsTales"),
    ("2148", "EdgarAllenPoe"),
    ("36", "WarOfTheWorlds"),
    ("829", "GulliversTravels"),
]

# Non-English Latin-script books, for Part 2.
LATIN_BOOKS = [
    ("2000", "Spanish", "es"),
    ("799", "French", "fr"),
    ("17489", "German", "de"),
    ("1012", "Italian", "it"),
    ("2650", "Portuguese", "pt"),
    ("7000", "Finnish", "fi"),
]

# --- Indic script: Wikipedia --------------------------------------------------
WIKI_LANGS = [("te", "Telugu"), ("hi", "Hindi"), ("ta", "Tamil")]


def _get_polite(url: str, params: dict, tries: int = 6):
    """GET with a fixed delay and exponential backoff on 429.

    Hammering the Wikipedia API with back-to-back requests earns a 429 within about
    twenty calls. A third of a second between requests is enough to stay under the
    limit, and the backoff covers the case where someone else on the IP is also busy.
    """
    delay = 2.0
    for attempt in range(tries):
        time.sleep(0.35)
        r = requests.get(url, params=params, headers=UA, timeout=60)
        if r.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"still rate-limited after {tries} attempts: {url}")


def normalise_newlines(text: str) -> str:
    """Collapse CRLF and lone CR to LF.

    This is not cosmetic. Several Gutenberg files contain thousands of *lone* CR
    characters, and Python's universal-newline handling rewrites them to LF when the
    cache is read back. Without normalising at fetch time, the first run (which
    tokenizes the freshly downloaded string) and every later run (which reads the
    cache) see different text and produce different token counts -- for The Great
    Gatsby, 84,246 tokens versus 82,535, a 2% gap at an identical character count,
    because GPT-2 tokenizes CR differently from LF. Normalising here makes a cold
    cache and a warm cache agree.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _cached(name: str, fetch) -> str:
    """Return cached text, fetching once if the cache is cold."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if path.exists():
        return normalise_newlines(path.read_text(encoding="utf-8"))
    text = normalise_newlines(fetch())
    # newline="" stops Windows translating LF back to CRLF on the way out.
    path.write_text(text, encoding="utf-8", newline="")
    return text


def strip_gutenberg_boilerplate(raw: str) -> str:
    """Drop the licence header and footer, which are English in every book.

    Leaving them in would contaminate the Spanish and Finnish samples with several
    hundred words of English -- small, but it biases exactly the statistic we measure.
    """
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw, re.S)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw, re.S)
    body = raw[start.end() if start else 0 : end.start() if end else len(raw)]
    return body.strip()


def gutenberg(code: str, title: str) -> str:
    def fetch() -> str:
        url = f"https://www.gutenberg.org/cache/epub/{code}/pg{code}.txt"
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        r.encoding = "utf-8"
        return strip_gutenberg_boilerplate(r.text)

    return _cached(f"gutenberg_{code}_{title}.txt", fetch)


def wikipedia(lang: str, target_chars: int = 150_000, max_requests: int = 60) -> str:
    """Concatenate plain-text article intros until `target_chars` is reached.

    Two things the API does that are easy to get wrong:

    * `exlimit` is silently capped at 1 unless `exintro` is set, so asking for full
      article text returns ONE extract per request no matter what limit you pass.
      That is why this uses intros: 20 per request instead of 1.
    * Random draws are dominated by stubs -- most te/hi/ta articles are two lines --
      so short extracts are filtered out and requests keep going until the target is
      met. Without the filter, the "Hindi sample" was 216 characters.
    """

    def fetch() -> str:
        chunks: list[str] = []
        total = 0
        for _ in range(max_requests):
            if total >= target_chars:
                break
            r = _get_polite(
                f"https://{lang}.wikipedia.org/w/api.php",
                {
                    "action": "query", "generator": "random", "grnnamespace": 0,
                    "grnlimit": 20, "prop": "extracts", "explaintext": 1,
                    "exintro": 1, "exlimit": 20, "format": "json",
                },
            )
            for p in r.json().get("query", {}).get("pages", {}).values():
                ex = (p.get("extract") or "").strip()
                if len(ex) >= 250:
                    chunks.append(ex)
                    total += len(ex)
        return "\n\n".join(chunks)

    return _cached(f"wikipedia_{lang}.txt", fetch)


def multilingual_corpus() -> dict[str, dict]:
    """{name: {text, script, source}} for every language in Part 2."""
    corpus: dict[str, dict] = {}

    corpus["English"] = dict(
        text=gutenberg("84", "Frankenstein"), script="Latin", source="Gutenberg"
    )
    for code, name, _ in LATIN_BOOKS:
        corpus[name] = dict(text=gutenberg(code, name), script="Latin", source="Gutenberg")

    for lang, name in WIKI_LANGS:
        corpus[name] = dict(text=wikipedia(lang), script="Indic", source="Wikipedia")

    return corpus


if __name__ == "__main__":
    c = multilingual_corpus()
    for name, d in sorted(c.items(), key=lambda kv: -len(kv[1]["text"])):
        print(f"  {name:<12} {d['script']:<6} {len(d['text']):>9,d} chars  ({d['source']})")
    print(f"\n  cached in {CACHE}")
