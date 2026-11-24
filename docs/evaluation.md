# Retrieval evaluation

Known-item benchmark: each natural-language question points to one paper. Two query sets, so that
conclusions do not rest on one kind of question (protocol and rules in
[`eval/README.md`](../eval/README.md)):

* **famous**: 60 well-cited papers (709 to 24k citations), including confusable pairs such as
  PYTHIA 6.4 / 8.1 / 8.2 and Planck 2015 / 2018;
* **longtail**: 40 papers with 20 to 200 citations sampled at random with a fixed seed, one question
  each written from the abstract under fixed rules (no three consecutive title words, checked by script).

Metrics: recall at k and MRR@10 with 95% bootstrap intervals; systems are compared with paired
sign-flip permutation tests. Per-question ranks are in `eval/results/`.

```bash
uv run --extra eval python eval/run_eval.py      # about 35 min on a laptop CPU the first time
```

## Live INSPIRE, whole database, 50 candidates

| system | famous MRR@10 [95% CI] | famous R@10 | longtail MRR@10 [95% CI] | longtail R@1 | longtail R@10 |
|---|---|---|---|---|---|
| raw question | 0.12 [0.05, 0.20] | 0.13 | 0.08 [0.01, 0.18] | 0.07 | 0.10 |
| OR query, INSPIRE ranking | 0.32 [0.22, 0.43] | 0.45 | 0.49 [0.35, 0.63] | 0.38 | 0.72 |
| OR + dense + rrf | 0.34 [0.24, 0.45] | 0.45 | 0.59 [0.46, 0.72] | 0.50 | 0.85 |
| **OR + cross-encoder (default)** | 0.30 [0.20, 0.41] | 0.43 | **0.76** [0.63, 0.87] | **0.70** | 0.85 |
| influential pool + cross-encoder | 0.36 [0.26, 0.46] | 0.55 | 0.73 [0.59, 0.85] | 0.68 | 0.80 |
| influential pool + dense | 0.36 [0.27, 0.47] | 0.58 | 0.56 [0.43, 0.70] | 0.45 | 0.78 |
| multi-query pool + dense + rrf | 0.34 [0.24, 0.45] | 0.57 | 0.49 [0.36, 0.62] | 0.38 | 0.75 |

Share of questions whose paper is among the 50 candidates:

| pool | famous | longtail |
|---|---|---|
| OR query | 0.53 | 0.88 |
| multi-query (OR + quoted phrases + title-only, interleaved) | 0.58 | 0.80 |
| influential (half from 250+ citations) | **0.75** | 0.83 |

Median reranking time for 50 new candidates on a laptop CPU: 2.6 to 2.8 s for the cross-encoder,
3.3 s for embeddings (46 ms when their vectors are already cached).

## Offline: fixed corpus of every INSPIRE record with 1500+ citations (1396 papers, famous set)

| system | R@1 | R@10 | MRR@10 [95% CI] |
|---|---|---|---|
| bm25 | 0.64 | 0.93 | 0.75 [0.66, 0.84] |
| dense (bge-small) | 0.74 | 0.98 | 0.82 [0.74, 0.90] |
| hybrid (rrf) | 0.74 | 0.97 | 0.83 [0.75, 0.90] |
| dense top 50, then cross-encoder | **0.86** | 0.98 | **0.90** [0.82, 0.96] |

## Findings

* **What limits retrieval depends on the question, and a one-sided benchmark gave the wrong answer.**
  With the famous set alone, no reranker beat INSPIRE's order (all p > 0.25) and candidate generation
  looked like the only lever. On typical papers the candidates are already there (88% of the time) and
  reranking is the lever: embeddings lift MRR from 0.49 to 0.59 (p = 0.014) and the cross-encoder to
  0.76 (p = 0.004 against embeddings), with the right paper first 70% of the time instead of 38%.
* **For famous papers the candidate pool is the bottleneck.** Their questions match thousands of
  follow-up papers. Reserving half of the pool to well-cited papers raises pool recall from 0.53 to
  0.75 (p < 0.001) and does not hurt the longtail set (p = 0.50), so it is exposed as
  `prefer_influential`, chosen by the model from the user's intent.
* **The cross-encoder is the default**: best on typical papers, no significant loss on famous ones,
  and no slower than embeddings once the candidates are new, which is the normal case for a live query.
  On the offline corpus it also lifts R@1 from 0.74 to 0.86 (MRR p = 0.02).
* **It also over-rewards literal matches.** Asked for "the paper that solved the hierarchy problem with
  a warped extra dimension", it ranks a 2018 paper titled almost like the question above Randall and
  Sundrum's original. Questions about famous papers remain the weak spot.
* **Negative results, kept on purpose:** the multi-query pool (adding quoted phrases and title-only
  queries) brings no significant gain on famous papers and loses on typical ones (MRR 0.49 vs 0.59,
  p = 0.004), because it fills the pool with noise; keeping only the rarest words of the question as
  an AND query finds the paper in 23% of cases.

## Limits

* One author wrote all questions knowing the targets; 37% of the famous-set questions repeat three
  consecutive title words (written before the rule existed), against 0% for the longtail set.
* 40 to 60 questions per set give wide intervals; only large differences are significant.
* Known-item search measures finding one paper, not topical recall or answer quality.
* The cross-encoder is a general web-search model (MS MARCO), not trained on physics.

## Next steps

* An LLM baseline that writes INSPIRE syntax through the MCP tools, scored with this same harness.
* Questions from real usage logs, and a larger longtail set to narrow the intervals.
* Physics-tuned rerankers compared on quality, latency and memory.
