# Prompt — Project 3 of 50 (10 August 2026)

## Role and place in the ladder

You are working in `50 ML Projects To Understand LLMs/50-llm-projects`, continuing the daily
ladder through *50 ML Projects To Understand LLMs* (Mike X Cohen). Projects 1 and 2 are done,
tested and pushed to `github.com/sravanni369/50-llm-projects`; `PROGRESS.md` is the log.

- **Project 1** found that Telugu costs GPT-2 roughly 14x more tokens than the same sentence in
  English.
- **Project 2** found that English household documents run about 6.3 characters per token, and
  that chunk retrieval beats pasting the whole document by 72.9%.

Project 3 is where those two facts get explained rather than just measured.

## The book concept

`ML4LLM_book/chapter_2/ml4llm_ch2_proj3_helper.ipynb` — **pandas frequency tables of token
lengths**. Tokenize with the GPT-2 BPE tokenizer, measure each token's length in characters,
build a frequency table, and plot the distribution: first for a single passage, then across ten
Project Gutenberg books on a log-x, frequency-normalised scatter.

## What to build

**Part 1 — the book exercise, done from scratch.** Reproduce both of the author's figures by
writing the code yourself rather than filling in his blanks: the token-length bar chart for the
Pulcinella passage, and the ten-book normalised log-x scatter. Cache every download so the
script re-runs offline.

**Part 2 — the problem worth solving.** Before sending a document to a pay-per-token API you
want to know two things: what script is it in, and is it going to be token-expensive? Sending
it somewhere to find out defeats the purpose on both cost and privacy. A token-length histogram
is a ~20-number fingerprint computable locally in milliseconds. Build that fingerprint across a
multilingual corpus spanning Latin-script languages (Gutenberg, public domain) and Indic scripts
including Telugu and Hindi (Wikipedia extracts), and show how the distribution shift explains the
cost multiplier that Project 1 measured.

**Part 3 — PyTorch.** Train a small classifier that predicts the script family from the
token-length histogram alone, and a regressor that predicts the chars-per-token cost multiplier.
Hold out documents the model has never seen. Report accuracy, a confusion matrix, and honest
error bars — if it does not work, say so.

## Constraints

- Read the concept from the author's MIT-licensed companion notebook only. **Reproduce no book
  text.** Code and results only, exactly as in Projects 1 and 2.
- Everything runs in VS Code from the repo folder. Every figure saved as a PNG.
- Tests in `test_project03.py`; all must pass before pushing.
- Screenshot the actual VS Code run into `screenshots/` using `scripts/screenshot_vscode.py`
  (window-render capture, side panel cropped — never a full-desktop grab).
- Commit and push as her, using `215950756+sravanni369@users.noreply.github.com`, so the
  contribution registers on her GitHub profile.
- Add the Project 3 row to `PROGRESS.md`.

## On the book-to-skill idea

Building a queryable skill from a book is a good fit for this ladder — but the output stays
**local**, in `~/.claude/skills/`, and is never pushed to the public repo. The ML4LLM book is a
DRM-protected Kindle purchase; only the free TOC/chapter-1 PDF and the MIT-licensed companion
notebooks exist locally. A skill built from owned book content is a private study aid, and
publishing it would redistribute copyrighted text. This repo's own README already commits to
"no book text is reproduced here" — keep that promise.
