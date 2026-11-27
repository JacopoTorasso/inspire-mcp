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

## Citation-context questions: how a physicist describes a paper from memory

The longtail and held-out questions were written from the abstract, which flatters a reranker that reads
the abstract (see Limits). A citing author describes a paper in their own words, so the sentence in which
another paper cites the target is a more realistic question, with a known answer.

`eval/citation_contexts.py` samples targets from the longtail population, downloads the LaTeX source of
papers citing them (no self-citations), and keeps the sentence around a `\cite` of the target when it
cites nothing else, names no author of the target, repeats no three consecutive title words and has no
equation; formulas become plain text as a user would type them (`$\Xi_c(2930)$` -> `Xi_c(2930)`).
`eval/judge_contexts.py` then asks Claude Haiku, which sees the sentence only, never the abstract,
whether it describes what the cited paper did specifically enough to pick it out; about one sentence in
three passes. One sentence per target is kept, by a fixed rule (`eval/freeze_citations.py`). Only ids
and hashes are versioned; the sentences stay in `eval/cache/`.

* **citations** (test, frozen before any model ran on it): 86 questions, two batches of 300 and 500
  sampled targets. Median share of a question's content words found in the target abstract: 0.27,
  against 0.75 for the longtail questions.
* **citedev** (development): 64 questions on other targets, same rules, for every choice below.

On these questions the default pipeline is much weaker than the longtail numbers suggest:

| development, 50 candidates | citedev MRR@10 | target in the pool |
|---|---|---|
| INSPIRE order of the OR query | 0.14 | 0.36 |
| MiniLM-L6 cross-encoder (default up to 0.5.3) | 0.10 | 0.36 |
| bge-reranker-base | 0.16 | 0.36 |
| Claude Haiku reranking the same 50 (`eval/llm_rerank.py`), reference only | 0.21 | 0.36 |
| the same, with the target inserted into every pool | 0.42 | 1 |

Two things limit the result: the target is missing from the pool for two questions in three (half of
them are not even in the first 250 results of the OR query), and a general web-search cross-encoder
reads a citing sentence poorly even when the target is there (MRR 0.13 with the target inserted, against
0.42 for Haiku).

**Candidates.** A long question starts with generic words, and the OR of its first twelve content words
can miss specific ones. Adding the OR of the twelve rarest words (document frequencies over 125k training
abstracts) raised the share of citedev targets in the pool from 0.36 to 0.41, with no loss on famous or
longtail (`eval/recall_lab.py`), but changed nothing on the test sets (citations 35/86 and heldout 34/40
either way), so it was not kept. AND queries of the rarest words, INSPIRE full-text search for them,
rarity measured on citing prose, and pools of 100 candidates did not help either (the rerankers rank a
larger pool worse). The rest of the gap needs either a semantic index of all of INSPIRE, which this
project avoids hosting, or a reformulated question, which is what an assistant calling the tool
provides: a Claude Haiku agent with semantic_search reaches MRR 0.22 on citedev (0.20 without it, 0.13
for a single call; `eval/agent_eval.py --sets citedev`).

**BM25 prefilter.** Scoring only the 25 candidates BM25 ranks highest halves the cross-encoder time with
no loss on any development set, citation questions included, for the general and the fine-tuned models,
and none on the test sets (MiniLM-L6: citations 0.18 and heldout 0.66 MRR@10 with or without it). It is
now the default.

**Fine-tuning** (`train/`): 3000 training papers sampled from the same populations, none in any evaluation
set; questions written by Haiku from the abstract and filtered by vocabulary overlap (`make_data.py`),
real citation sentences from Semantic Scholar judged as above (`s2_contexts.py`), hard negatives from the
INSPIRE pools the server would rerank, and Haiku's rankings of those pools distilled into the student
(`teacher.py`, `finetune.py`). Results and the model choice: see the section below, written once the
test set has been run.

**Choice, fixed before the test sets were run (26/11/2026).** On the development sets, by mean MRR@10
with the target inserted and with no set worse than the default on the real pools, the fine-tuned
bge-reranker-base with distillation wins (0.524; real pools 0.424 against 0.396 for MiniLM-L6, better
on all three sets), and the distilled MiniLM-L12 is the fast alternative (0.505; 0.421). Both are run
once on the test sets (`citations`, then `heldout`), together with the default; the test results do not
choose between them: the bge model becomes the default if a complete semantic_search stays within 3.5 s
on a laptop CPU, the L12 model otherwise.

### Test results: the fine-tuned models are not adopted

Run once, on the frozen test sets, with the server's pipeline (50 candidates, BM25 prefilter, ONNX
through fastembed; the bge model quantized to int8 in its feed-forward layers):

| test, MRR@10 [95% CI] | citations (86) | heldout (40) |
|---|---|---|
| MiniLM-L6 (default) | 0.18 [0.11, 0.25] | 0.66 [0.53, 0.79] |
| bge-reranker-base fine-tuned, distilled | 0.21 [0.13, 0.29], p = 0.38 | 0.58 [0.45, 0.71], p = 0.06 |
| MiniLM-L12 fine-tuned, distilled | 0.16 [0.10, 0.23], p = 0.36 | 0.65 [0.51, 0.78], p = 0.86 |

Neither model beats the default: the bge model gains a little on citation questions, not
significantly, and loses on the held-out set; the L12 model is level on the held-out set and slightly
worse on citations. The development gains (0.396 -> 0.424 on the real pools) did not carry over: many
variants were compared on 164 development questions, and the best of them on those questions is also
the one most favoured by chance. The default stays MiniLM-L6. What the work did establish: questions
from citing sentences are much harder than the abstract-written ones (MRR 0.18 against 0.66), the main
limit on them is the candidate pool, and an LLM reranker (Haiku) is far ahead of every cross-encoder
when the target is in the pool (0.42 against 0.13 to 0.28 on citedev), so better training data, not a
bigger cross-encoder, is where the next gain would come from.

## Dense index (service option): development results and the test fixed in advance

`inspire-mcp index build DIR` embeds the 836k INSPIRE records with an arXiv abstract (bge-small, fastembed);
with `--dense-index DIR` half of semantic_search's candidates come from the records nearest to the question
by meaning, half from INSPIRE. On the development sets, with the questions an assistant actually passes to
the tool (`eval/agent_queries.py`: Claude rewrites every description, in about 12 words), the share of
targets among the 50 candidates rose from 0.39 to 0.50 (citedev), 0.61 to 0.82 (famous) and 0.93 to 0.97
(longtail). The cross-encoder turns little of it into a better first place, but the assistant does: the
final ranking of a Claude Haiku agent improved from MRR 0.252 to 0.315 on citedev (p = 0.06) and from
0.815 to 0.864 on longtail (p = 0.02), averaged over two or three runs per configuration (agent runs vary
by about 0.05); famous papers were already found (0.99).

**Test, fixed before it was built (26/11/2026).** The citation test set has been used, so a fresh one is
drawn with the same pipeline (`citations2`: seed 2031, 500 sampled targets, none in any earlier batch,
development set or training set). It is frozen before any run, then the Haiku agent runs twice with and
twice without the index. The index is reported as an improvement if the per-question MRR averaged over
the runs is higher with it, with a paired permutation p < 0.05; otherwise it is reported as not confirmed.

**Test result: not confirmed.** On `citations2` (62 questions; fewer than hoped, since 62 of the 500
sampled targets had a sentence the judge kept) the agent's MRR@10 was 0.287 without the index and 0.296 with
it, averaged over two runs each: +0.009, paired permutation p = 0.79 (9 questions better, 12 worse, 41
equal); a single semantic_search call scored 0.20 without and 0.17 with it. The development gain did not
carry over to new questions: the index is kept as an experimental option, off by default, and is not
described as an improvement. More candidates in the pool are not enough: neither the cross-encoder nor the
agent turned them into better answers on new questions.

## Agent efficiency: can smaller tool output make an assistant cheaper?

The question here is not accuracy but what an assistant spends to get it. `eval/agent_eval.py --mcp` has a
Claude Haiku agent use the real server in process (its 18 tools, their descriptions, the instructions, and
results exactly as an MCP client receives them), on the 164 development questions (famous, longtail,
citedev). **Criterion, fixed before any variant (26/11/2026):** a change is kept if tool calls per question
drop by 20% or cost per question (a proxy for tokens) by 25%, with pooled MRR@10 down by at most 0.03, two
runs per configuration. `eval/compare_runs.py` applies it: per-question means over the runs, paired
permutation p, bootstrap 95% intervals over questions.

Baseline (two runs): MRR 0.701, 3.67 tool calls and $0.00299 per question. Each variant is two runs,
compared with the baseline; variants 3 to 6 build on the one before.

| variant | calls | cost | MRR [95% CI] | citedev MRR |
|---|---|---|---|---|
| 1. instructions: results are ranked, answer when one fits | -11.0% | +1.0% | -0.003 [-0.021, +0.014] | 0.335 |
| 2. semantic_search: whole abstract for the first 3, 400 characters after; no keywords, categories, pdf_url | -1.3% | -20.1% | -0.008 [-0.027, +0.009] | 0.317 |
| 3. + compact JSON text instead of indented | +4.8% | -21.3% | -0.010 [-0.033, +0.010] | 0.314 |
| 4. + results after the first 3 as light records (first author, no journal or DOI) | +2.6% | -22.8% | -0.008 [-0.028, +0.010] | 0.311 |
| 5. + tool schemas without pydantic's per-argument titles | +2.6% | **-24.6%** [-28.9%, -20.2%] | -0.004 [-0.022, +0.013] | 0.328 |
| 6. + 250-character excerpts instead of 400 (declared the last) | +5.5% | -24.5% | -0.010 [-0.032, +0.008] | 0.304 |

Baseline citedev MRR: 0.347. **No variant meets the criterion.** The best, variant 5, misses the cost
threshold by less than half a point, with no measurable loss of accuracy (the MRR interval stays within the
allowed 0.03). **Variant 5 is adopted (27/11/2026) by a decision taken after seeing the result**, not by the
criterion: a quarter less cost per question for an assistant, nothing measurable lost. Variant 6 is not
adopted. On citedev every variant scored below the baseline (not significant), so that set is the one to
watch. What the measurements show:

* **semantic_search produces most of what the agent reads**: 81% of the characters of tool results in the
  baseline runs (about 18,000 per call, half of them abstracts), against 18% for search_literature.
* **Characters are not tokens.** Physics abstracts tokenize at about 2.8 characters per token, and
  whitespace is nearly free: removing JSON indentation cut 12% of the characters and 1.6% of the cost.
* **A fixed part caps the saving.** Tools and instructions are about 5,000 tokens (4,300 without the
  titles), written to the prompt cache once per conversation: about $0.0006, some 40% of the cost of a
  famous-set question, which no change to results reaches.
* **Less information is paid back in calls.** Shorter results did not lower the number of tool calls: from
  variant 3 on the agent made a few percent more, and with 250-character excerpts the extra calls cancelled
  the saving. On citedev, where the agent searches longest, every variant scored below the baseline
  (0.304 to 0.335 against 0.347, none significant), consistent with the full abstracts helping there.

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
* Longtail and held-out questions were written from the abstract, which is also the text the reranker
  reads, so they share its vocabulary more than a user describing a paper from memory would. On
  average 75% of a question's content words appear in the target abstract, against 47% for the other
  candidates, and counting shared words alone puts the target first for 13 to 21 of 40 questions on
  the development set and 13 to 18 on the held-out set, depending on ties, against 28 and 24 for the
  cross-encoder (`eval/abstract_overlap.py`). Real-world accuracy is therefore likely lower than
  measured here.
  Split by overlap, the cross-encoder puts the right paper first for 9 of 23 questions sharing less
  than 60% of their words with the abstract, against 27 of 33 above 80% (all three sets pooled,
  targets in the pool): it works best exactly when a question echoes the abstract.
* 40 to 60 questions per set give wide intervals; only large differences are significant.
* Known-item search measures finding one paper, not topical recall or answer quality.
* The cross-encoder is a general web-search model (MS MARCO), not trained on physics.

## Next steps

* A reranker fine-tuned on physics text: tried (see the citation-context section), it beat the default
  on the development sets but not on the test sets. A next attempt needs more real citation sentences
  for training and a larger development set, so that choosing among variants does not fit noise.
* Candidate recall on citation-like questions (a third to two fifths of the targets reach the pool):
  no cheap query rewriting tried so far helps.
* Questions from real usage logs, and a larger longtail set to narrow the intervals.
* Physics-tuned rerankers compared on quality, latency and memory.
