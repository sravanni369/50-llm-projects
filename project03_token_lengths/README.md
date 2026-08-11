# Project 03: Token-length histograms, and what they tell you before you spend a rupee

Book concept ([ML4LLM](https://github.com/mikexcohen/ML4LLM_book) project 3): build **pandas
frequency tables of token lengths** — tokenize with GPT-2 BPE, measure each token's length in
characters, and study the distribution rather than the mean.

Applied here to the question Projects 1 and 2 left open. Project 1 found Telugu costs GPT-2 far
more tokens than English; Project 2 found English documents run about 6.3 characters per token.
Neither explained *why*. The distribution does.

The practical use: before sending a document to a pay-per-token API you want to know what script
it is in and what it will cost. Sending it somewhere to find out defeats the purpose on both
privacy and cost. A token-length histogram is 20 numbers you can compute locally in milliseconds.

## Part 1: the book exercise

One paragraph, then ten Project Gutenberg books normalised and plotted on a log-x axis.

![passage](part1_passage.png)

The passage tokenizes to 172 tokens from 677 characters. Token lengths run 1–13 characters and
peak at 4.

![books](part1_books.png)

| | chars/token |
|---|---|
| Romeo & Juliet | 2.89 |
| Alice in Wonderland | 3.08 |
| Huckleberry Finn | 3.09 |
| Edgar Allen Poe | 3.18 |
| The Great Gatsby | 3.29 |
| Grimms' Tales | 3.45 |
| Heart of Darkness | 3.61 |
| War of the Worlds | 3.76 |
| Gulliver's Travels | 3.86 |
| Frankenstein | 3.94 |

Ten books across two centuries, in registers from Elizabethan verse to Mississippi vernacular,
span **1.06 chars/token end to end**. The tokenizer's vocabulary, not the author, decides where
words break. (Romeo & Juliet sits at the bottom because a play is full of speaker labels and
line breaks, and a newline is a token.)

## Part 2: nine languages, two shapes

Latin-script books from Gutenberg; Telugu, Hindi and Tamil from Wikipedia extracts (~150k
characters each, balanced on purpose).

![cost](part2_language_cost.png)

| language | script | chars/token | length-1 tokens | cost vs English |
|---|---|---|---|---|
| English | Latin | 3.94 | 22.5% | 1.0x |
| French | Latin | 2.57 | 26.9% | 1.5x |
| Spanish | Latin | 2.53 | 21.8% | 1.6x |
| German | Latin | 2.49 | 25.1% | 1.6x |
| Portuguese | Latin | 2.48 | 22.4% | 1.6x |
| Italian | Latin | 2.32 | 25.8% | 1.7x |
| Finnish | Latin | 2.07 | 25.3% | 1.9x |
| **Hindi** | Indic | 0.67 | **49.0%** | **5.9x** |
| **Telugu** | Indic | 0.39 | **99.2%** | **10.1x** |
| **Tamil** | Indic | 0.39 | **99.2%** | **10.2x** |

**What "token length" means here.** GPT-2 is a *byte-level* BPE tokenizer — it never sees
characters, only UTF-8 bytes mapped into a printable surrogate alphabet. For English one
surrogate is one letter. For Telugu, where one character is three UTF-8 bytes, it is not. So the
histogram measures bytes per token, while chars/token is measured against real user-visible
characters. The gap between those two numbers *is* the cost story.

Telugu and Tamil are at **99.2% single-byte tokens**: GPT-2's vocabulary contains essentially no
merges for either script, so every byte becomes its own token and the tokenizer degenerates to
raw UTF-8. Hindi sits at 49% — Devanagari appears often enough in GPT-2's training data to have
earned some merges. That difference is worth 4 percentage points of cost multiplier.

Even Finnish, in the same alphabet as English, costs 1.9x — agglutinative morphology produces
long words the vocabulary has never seen.

## Part 3: PyTorch on 20 numbers

Two small MLPs over the same histogram. 76 windows of 2,000 characters per language (balanced),
split contiguously 70/30 so held-out text comes from a different part of each document.

![predict](part3_predict.png)

**Task 1 — which language?** 78.3% on 9 held-out classes against a 10% chance baseline. Script
(Latin vs Indic), which is the question that actually sets the bill, is **100%**.

**Task 2 — what will it cost?** MAE **0.022 chars/token** (1.4% MAPE), against 0.885 for
predicting the training mean — **39x better**. On nine English books the model never saw,
9/9 identified correctly with cost predicted to within 0.07 chars/token.

### Where it fails, and why that is the interesting part

Telugu scores 100% and Tamil 0%. Not a bug, and not fixable by training longer:

| pair | L1 distance between histograms |
|---|---|
| Tamil vs Telugu | **0.0020** |
| German vs Portuguese | 0.0541 |
| French vs German | 0.0865 |

The two closest non-Indic languages are **27x further apart** than Telugu and Tamil. Both scripts
collapse to pure byte tokens, so their fingerprints coincide and the classifier can only pick one.
Which one it picks is a coin flip decided by the initialisation seed:

| seed | overall | Telugu | Tamil | script |
|---|---|---|---|---|
| 0 | 78.3% | 100% | 0% | 100% |
| 1 | 78.7% | 0% | 100% | 100% |
| 2 | 78.3% | 100% | 0% | 100% |
| 3 | 78.7% | 100% | 0% | 100% |
| 4 | 77.0% | 0% | 100% | 100% |
| 5 | 80.0% | 100% | 0% | 100% |

Honest read: the histogram is a **cost** feature, not a **language-ID** feature. It nails the
thing you would actually use it for — 100% script detection and 1.4% cost error, every seed — and
it cannot separate two languages the tokenizer treats identically. Telling Telugu from Tamil needs
a feature that looks at *which* bytes, not how many.

## Run it

```bash
pip install torch transformers pandas seaborn matplotlib pytest
python corpus.py                 # downloads + caches; everything after this is offline
python part1_book_exercise.py
python part2_language_cost.py
python part3_predict.py
python -m pytest test_project03.py -q
```

All 16 tests pass. The full run takes a few minutes on first execution while the corpus downloads;
everything after that is offline.

## Files

| file | what it does |
|---|---|
| `corpus.py` | fetch + cache Gutenberg books and Wikipedia extracts |
| `fingerprint.py` | the 20-number feature, defined once |
| `part1_book_exercise.py` | the book's two figures |
| `part2_language_cost.py` | nine-language table and cost multipliers |
| `model.py` | the two PyTorch heads |
| `part3_predict.py` | training, held-out evaluation, unseen-document check |
| `test_project03.py` | 16 tests |
| `PROMPT.md` | the prompt this project was built from |

Two API traps worth knowing, both of which cost a debugging cycle here and are commented in
`corpus.py`: MediaWiki silently caps `exlimit` to 1 unless `exintro` is set, so a request for 20
full extracts returns one; and random Wikipedia draws in te/hi/ta are overwhelmingly stubs, which
is how the first "Hindi corpus" came back at 216 characters.

Source text is cached under `cache/` and not committed — only derived statistics are.
No book text is reproduced here.
