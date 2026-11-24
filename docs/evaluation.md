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

## Agentic baseline: does a capable LLM need semantic_search?

An assistant is not a naive user: it knows INSPIRE syntax and can reformulate. `eval/agent_eval.py` runs
the same 100 questions through an agent loop (effort `medium`, at most 8 turns, final answer through a
`submit_ranking` tool) in two arms: **A**, the agent has only `search_literature`; **B**, it also has
`semantic_search`. Single-call `semantic_search` without any LLM is the reference.

Longtail set (40 randomly sampled papers):

| system | MRR@10 [95% CI] | right paper first | tool calls per question | cost per question |
|---|---|---|---|---|
| `semantic_search`, one call, no LLM | 0.76 [0.63, 0.87] | 70% | 1 | none |
| Claude Sonnet 5.5, INSPIRE search only | 0.60 [0.45, 0.75] | 60% | 5.8 (9 of 40 ran out of turns) | $0.016 |
| **Claude Sonnet 5.5 + semantic_search** | **0.86** [0.75, 0.95] | **85%** | 1.3 | $0.010 |
| Claude Haiku 5.5, INSPIRE search only | 0.56 [0.40, 0.71] | 55% | 8.2 | $0.0016 |
| **Claude Haiku 5.5 + semantic_search** | **0.83** [0.72, 0.93] | **80%** | 2.7 | $0.0008 |

Costs at Anthropic list prices of November 2026; the whole run cost $2.05.

* **semantic_search makes both agents better, faster and cheaper**: MRR up 0.26 for Sonnet and 0.27 for
  Haiku (p < 0.001 for both), with four to five times fewer tool calls.
* **A small model with the tool beats a large model without it**: Haiku with semantic_search (0.83)
  is well above Sonnet with INSPIRE search alone (0.60), at a twentieth of the cost per question.
* **On the development set, an agent searching alone did worse than the tool alone** (0.60 and 0.56
  against 0.76; see the held-out check below for Sonnet). The
  transcripts show why: the agent looks for the description's words in titles (`t homotopy and t
  L-infinity and t string`) and keeps reformulating the same way, while the real titles share no word
  with the description ("Higher Structures in M-Theory"). semantic_search reads the abstracts.
* With the tool, the agent adds a gain over the tool alone (Sonnet 0.86 against 0.76, p = 0.03):
  it can read the candidates and check them against the question.

**Held-out confirmation** (the 40 papers frozen before any result, same protocol, cost $1.18):

| system | MRR@10 [95% CI] | right paper first | tool calls per question |
|---|---|---|---|
| question typed into INSPIRE (31 of 40 return nothing) | 0.12 | 12% | 1 |
| `semantic_search`, one call | 0.66 [0.52, 0.79] | 60% | 1 |
| Claude Sonnet 5.5, INSPIRE search only | 0.64 [0.49, 0.79] | 62% | 5.1 (8 of 40 ran out of turns) |
| **Claude Sonnet 5.5 + semantic_search** | **0.88** [0.78, 0.96] | **85%** | 1.5 |
| Claude Haiku 5.5, INSPIRE search only | 0.53 [0.38, 0.68] | 53% | 7.5 |
| **Claude Haiku 5.5 + semantic_search** | **0.84** [0.73, 0.94] | **82%** | 3.1 |

The agent results replicate: with semantic_search both agents gain (Sonnet p = 0.02, Haiku p = 0.001)
and beat the tool alone (p = 0.003 and 0.002). One development finding does not replicate: on held-out
papers Sonnet searching alone (0.64) is level with single-call semantic_search (0.66, p = 0.83), so
"an agent searching alone does worse than the tool alone" holds for Haiku, not for Sonnet.

On the famous set every agent configuration finds the paper almost always (MRR 0.99 to 1.00, against
0.30 for single-call semantic_search), because the model remembers these papers and searches for their
authors or titles directly. That measures the model's memory, not retrieval, so the longtail set is the
one that answers the question. The longtail questions avoid title words by design, which penalises
title searches; it is also the situation of a user who does not know the title.

## Can a different reranker do better? Comparison and held-out confirmation

`eval/rerank_lab.py` freezes 100 candidates per question and compares cross-encoders on exactly the
same candidates. Choices are made on the development sets (famous and longtail); the winner is then
checked once on a **held-out set** of 40 further long-tail papers (seed 2027, questions written and
committed before any result was seen, same rules).

Development sets, MRR@10 and right paper first:

| reranker | size | s per 50 candidates | famous MRR | longtail MRR | longtail first |
|---|---|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 (default) | 80 MB | 2.1 | 0.31 | 0.76 | 70% |
| ms-marco-MiniLM-L-12-v2 | 120 MB | 4.3 | 0.33 | 0.79 | 75% |
| jina-reranker-v1-tiny-en | 130 MB | 2.5 | 0.31 | 0.71 | 65% |
| jina-reranker-v1-turbo-en | 150 MB | 3.8 | 0.32 | 0.69 | 60% |
| bge-reranker-base | 1 GB | 14.3 | 0.25 | 0.72 | 65% |

MiniLM-L12 came out best on both development sets, but not significantly (p = 0.20 and 0.25).
Doubling the pool to 100 candidates, fusing cross-encoders, or adding embeddings brought nothing.

**Held-out confirmation (40 new papers):** MiniLM-L12 and MiniLM-L6 are equivalent (right paper first
60% for both, MRR 0.67 against 0.66, p = 0.82), against 40% for INSPIRE's own order. The development
gain was noise, so the default stays MiniLM-L6, which is twice as fast. Two lessons:

* the larger model was the worst: bge-reranker-base is seven times slower and loses on both sets;
* the held-out set is also the honest estimate of quality on unseen papers: 60% right first, lower
  than the 70% measured on the set the design choices were made on.

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

* A reranker fine-tuned on physics text: general web-search rerankers have plateaued here.
* Questions from real usage logs, and a larger longtail set to narrow the intervals.
* Physics-tuned rerankers compared on quality, latency and memory.
