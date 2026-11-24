# Evaluation protocol

Known-item retrieval: each question has exactly one target paper, identified by arXiv id and
resolved to its INSPIRE record at run time. A system succeeds at k when the target is among its
first k results.

## Query sets

| set | file | targets | how the questions were made |
|---|---|---|---|
| famous | `queries.jsonl` | 60 well-cited papers, 709 to 24k citations | written by the author, who knew the papers; several deliberately confusable pairs (PYTHIA 6.4 / 8.1 / 8.2, Planck 2015 / 2018, AdS/CFT paper / review) |
| longtail | `queries_longtail.jsonl` | 40 papers with 20 to 200 citations | targets sampled by `sample_longtail.py` (seed 2026, `topcite 20->200 and date 2015->2022 and tc p`, arXiv abstract of 400+ characters); one question per paper written from its abstract |

Rules for the longtail questions, checked by script before freezing the file:

* paraphrase of the main contribution, as a researcher would ask an assistant;
* at most 20 words;
* no three consecutive words copied from the title (12 of the first 40 drafts broke this rule and were rewritten).

`run_eval.py` reports for each set the share of questions that repeat three consecutive title words,
so the rule can be verified on every run. The famous set was written before the rule existed and
does not follow it.

## Metrics and statistics

* Recall at 1, 5 and 10, and MRR@10 (reciprocal rank of the target, 0 when outside the top 10).
* 95% confidence intervals: percentile bootstrap over questions, 2000 resamples.
* Comparisons between two systems: two-sided paired sign-flip permutation test on per-question
  MRR@10 (10,000 permutations); the same test on 0/1 outcomes compares candidate pool recall.
* Seeds are fixed (`SEED = 13`), so a rerun on the same INSPIRE data gives the same intervals.

## What changes between runs

The live part queries the production INSPIRE index: new papers, citation counts and ranking changes
move the numbers over time. That is intended, as the weekly workflow is a regression monitor. The
offline corpus is cached in `cache/corpus.json`; `--refresh` downloads a new snapshot.

## Known limits

* One author wrote all questions, knowing the targets.
* Known-item search measures finding one paper, not topical recall or answer quality.
* With 40 to 60 questions per set, one question moves a recall by 0.017 to 0.025, so only large
  differences reach significance.
