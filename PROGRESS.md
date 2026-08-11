# Progress

| # | Date | Concept | Problem chosen | Result |
|---|------|---------|----------------|--------|
| 1 | 2026-08-06 | Three tokenization schemes (char / word / GPT-2 BPE) | Book exercise + Telugu token-cost study + JD skill-term fragmentation | 9/9 tests pass; Telugu ≈14x GPT-2 token inflation on 3 sentence pairs (13.4–15.0x); PyTorch/scikit-learn shatter under BPE |
| 2 | 2026-08-07 | Doc lengths in chars/words/tokens; count correlations | Household paperwork (lease/EOB/benefits handbook) vs pay-per-token API costs; chunk-retrieval alternative; PyTorch chars→tokens estimator | 6/6 tests pass; full-doc vs chunked = 72.9% cost cut ($18.08 → $4.90/yr at 12 q/mo); linear fit MAE 2.2 tokens (4.8% MAPE), ≈6.3 chars/token |

| 3 | 2026-08-10 | Pandas frequency tables of token lengths | Token-length histogram as a local, private pre-flight check: what script is this, and what will it cost? 9 languages, Latin vs Indic; PyTorch classifier + cost regressor | 18/18 tests pass; Telugu/Tamil both 10.6x English token cost at 99.2% single-byte tokens, Hindi 6.1x; script detection 100% on every seed, 9-way language 77.8%; cost MAE 0.023 chars/token (1.6% MAPE, 39x over baseline); Telugu vs Tamil provably indistinguishable from the histogram (L1 0.0020, 34x closer than any other pair). Caught and fixed a CRLF cache bug that had made results depend on whether the cache was warm |

Next up: **Project 4** (`ML4LLM_book/chapter_2/ml4llm_ch2_proj4_helper.ipynb`).
