# Prompt — Project 4 of 50 (11 August 2026)

## Place in the ladder

`50-llm-projects`, continuing the daily run through *50 ML Projects To Understand LLMs*
(Mike X Cohen). Projects 1–3 are pushed; `PROGRESS.md` is the log. Yesterday (project 3) built
token-length histograms and trained two PyTorch heads on them.

## Book concept

`ML4LLM_book/chapter_2/ml4llm_ch2_proj4_helper.ipynb` — **token lengths in characters and
bytes**. Every GPT-2 token has two lengths: how many characters it decodes to, and how many
UTF-8 bytes those characters occupy. For ASCII they are equal. For anything else they are not.
`ü` is 2 bytes, `活` is 3, `🥰` is 4.

## The constraint

**The detector must be under 20 lines.** Legible code, no semicolon golf.

## The problem to aim it at

Children's education, and specifically who can afford an AI tutor.

Project 3 found that GPT-2 turns Telugu into 99.2% single-byte tokens and charges 10.6x English
per character. Project 4 explains *why* in one number: a Telugu character is 3 UTF-8 bytes, and
a byte-level tokenizer with no Telugu merges bills per byte. So a Telugu-medium child and an
English-medium child asking an AI tutor the same question do not pay the same price — the Telugu
child's family pays several times more for the identical lesson, or gets several times less
tutoring for the same budget.

That is a measurable equity gap, not a metaphor. Quantify it in rupees per lesson.

## The self-critical finding to chase

Yesterday I trained two MLPs, 760 samples, 600 epochs, to predict characters-per-token to 1.6%
error. Today's vocabulary table should give the *same* quantity **exactly**, with no training at
all, because the decoded character lengths of a text's tokens must sum to the length of the text.

If that identity holds, say so plainly: **yesterday's model was unnecessary.** Verify it to the
character on real text rather than asserting it.

## On the nature half

The second lever on children's attention is green space — Attention Restoration Theory (Kaplan,
1995) and Taylor & Kuo's finding that greener play settings reduce ADHD symptom severity.

**This project does not measure that.** No public dataset links Indian school green space to
attention outcomes, and inventing one would be worse than omitting it. Cite the literature as
framing, label it clearly as cited rather than computed, and keep it out of every results table.
The code measures the tokenizer half only. Say that in the README.

## Method framing (Low-Code AI, Stripling & Abel, O'Reilly 2023)

That book insists on a **use-case-first, data-first** order: state the goal and the question
before choosing a model, because "any ML project must begin with defining a goal, use case, or
problem." Follow it literally here. The goal is a rupee figure per lesson, not an accuracy
score, and today's hypothesis is falsifiable before any model exists. If the data says the
hypothesis is wrong, that is the finding — report it rather than reaching for a model to rescue
it.

## Who this is actually about (The Hindu, Vijayawada)

Ground the education claim in reported fact, not abstraction:

- Andhra Pradesh runs **tribal welfare residential schools and ashram schools**, with the Tribal
  Welfare Department stating it is working to improve their educational standards and
  infrastructure (10 August 2026).
- ₹4,764 crore allocated for the development of **27.39 lakh tribal people**; **1.5 lakh tribal
  youth** to get skill training and special support preparing for competitive examinations
  (10 August 2026).
- A student was **electrocuted at a tribal welfare school** in Velerupadu; ITDA and Tribal
  Department officials visited (8 August 2026).

These are the children on the wrong side of the tokenizer gap: Telugu-medium, state-funded, least
able to absorb a 10.6x price on the same lesson. Cite with dates. Do not invent enrolment or
outcome statistics that the paper does not report.

## Build

1. `vocab.py` — the under-20-line table and the exact cost function.
2. `part1_book_figure.py` — the author's characters-vs-bytes scatter across all 50,257 tokens.
3. `part2_tutor_cost.py` — cost per lesson, Telugu vs English, on real text.
4. `test_project04.py` — tests, including the exactness identity and a tokenizer round-trip.
5. Run in VS Code, screenshot the real terminal into `screenshots/`.
6. README with the numbers, the figure, and the honest split between measured and cited.
7. Commit as her, push, update `PROGRESS.md`.

## Truth rule

Every number from the actual run. Reuse the cached corpora from project 3 rather than
re-downloading. No invented statistics for the green-space section — citation only.
