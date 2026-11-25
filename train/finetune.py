"""Fine-tune the MS MARCO cross-encoder on physics questions, on the GPU, in the .venv-train environment.

Each training question (make_data.py) is one group: the target paper and N hard negatives drawn from the
INSPIRE pool the server would rerank for that question, scored together with a softmax cross-entropy
(listwise) loss, as rerankers are usually trained. With --teacher, questions ranked by train/teacher.py
use the 20 candidates the teacher saw instead, and the loss adds the KL divergence from the teacher's
ranking: a signal on every candidate, and no penalty on candidates the teacher finds relevant too.
Papers are encoded exactly as the server encodes them:
title, a full stop, the abstract, cut at 1200 characters (inspire_mcp.semantic.doc_text).

After each epoch the model reranks the frozen development pools of eval/rerank_lab.py (citedev, the
citation-context questions, and the older famous and longtail sets), as is and with the target paper
inserted into the pool (gold_docs.py); the epoch kept is the best on the mean of the three sets with the
target inserted, which judges the reranker on all 164 questions. The test sets are never read here.

    .venv-train/bin/python train/finetune.py --base cross-encoder/ms-marco-MiniLM-L6-v2 --out train/models/l6
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "train" / "cache"
EVAL_POOLS = (ROOT / "eval" / "cache" / "pools-citedev.json", ROOT / "eval" / "cache" / "pools-famous-longtail.json")


def eval_pools(tag: str) -> tuple[Path, ...]:
    return tuple(p.with_name(f"{p.stem}-{tag}.json") for p in EVAL_POOLS) if tag else EVAL_POOLS


MAX_CHARS = 1200  # inspire_mcp.semantic.doc_text
MAX_TOKENS = 320
# the epoch kept: best mean MRR@10 over the three development sets with the target inserted, so that no kind
# of question is traded for another (decided before the bge distillation results were seen)
SELECT = ("citedev+target", "famous+target", "longtail+target")


def selection(report: dict) -> float:
    return statistics.fmean(report[name]["MRR@10"] for name in SELECT)


def doc_text(p: dict) -> str:
    return f"{p['title']}. {p.get('abstract') or ''}"[:MAX_CHARS]


SOURCES = {"haiku": CACHE / "questions.jsonl", "s2": CACHE / "s2_judged.jsonl"}


Group = tuple[str, list[str], list[float] | None]  # question, docs (the target first), teacher distribution


def teacher_distribution(candidates: list[int], ranking: list[int], target: int) -> tuple[list[int], list[float]]:
    """Candidates reordered with the target first, and the teacher's probabilities: 1/(rank+1) over its ten
    choices, a small floor for the others."""
    order = [target, *(c for c in candidates if c != target)]
    weight = {r: 1.0 / (i + 1) for i, r in enumerate(ranking)}
    raw = [weight.get(c, 0.02) for c in order]
    total = sum(raw)
    return order, [w / total for w in raw]


def training_groups(
    sources: list[str], negatives: int, skip_head: int, teacher: dict, rng: random.Random
) -> list[Group]:
    papers = {p["recid"]: p for p in map(json.loads, (CACHE / "papers.jsonl").read_text().splitlines())}
    questions = {q["id"]: q for name in sources for q in map(json.loads, SOURCES[name].read_text().splitlines())}
    groups: list[Group] = []
    for line in (CACHE / "pools.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # the last line while make_data.py pools is still writing
            continue
        if row["id"] not in questions:
            continue
        query = questions[row["id"]]["query"]
        docs_by_id = {p["recid"]: p for p in row["pool"]} | {row["recid"]: papers[row["recid"]]}
        if row["id"] in teacher:  # the candidates the teacher ranked, with its distribution
            t = teacher[row["id"]]
            order, probs = teacher_distribution(t["candidates"], t["ranking"], row["recid"])
            groups.append((query, [doc_text(docs_by_id[c]) for c in order], probs))
            continue
        # the first candidates are the likeliest false negatives: papers that also answer a vague question
        pool = [p for p in row["pool"] if p["recid"] != row["recid"]][skip_head:]
        if len(pool) < negatives:
            continue
        # half from the head of the pool (what the reranker must beat), half from the rest
        head, tail = pool[:10], pool[10:]
        negs = rng.sample(head, min(len(head), negatives // 2))
        negs += rng.sample(tail, min(len(tail), negatives - len(negs)))
        groups.append((query, [doc_text(papers[row["recid"]]), *(doc_text(p) for p in negs)], None))
    return groups


@torch.no_grad()
def scores(model, tok, query: str, docs: list[str], device: str) -> list[float]:
    out = []
    for i in range(0, len(docs), 64):
        batch = tok(
            [query] * len(docs[i : i + 64]),
            docs[i : i + 64],
            truncation="only_second",
            max_length=MAX_TOKENS,
            padding=True,
            return_tensors="pt",
        ).to(device)
        out += model(**batch).logits[:, 0].float().tolist()
    return out


def evaluate(
    model, tok, device: str, pools: tuple[Path, ...] = EVAL_POOLS, deep: bool = False
) -> dict[str, dict[str, float]]:
    """R@1 and MRR@10 on each development pool as the server sees it, and with the target inserted
    (replacing the last candidate when missing): the second measures the reranker alone, on every question."""
    gold = json.loads((CACHE / "gold_docs.json").read_text())
    model.eval()
    report = {}
    variants = ("pool", "+target", "pool@100") if deep else ("pool", "+target")
    for path in pools:
        for set_name, queries in json.loads(path.read_text()).items():
            for variant in variants:
                pool_size = 100 if variant == "pool@100" else 50
                r1, mrr = [], []
                for q in queries:
                    pool = q["pool"][:pool_size]
                    if variant == "+target" and q["recid"] not in {p["recid"] for p in pool}:
                        pool = [*pool[: pool_size - 1], gold[str(q["recid"])]]
                    s = scores(model, tok, q["query"], [doc_text(p) for p in pool], device)
                    ranked = [p["recid"] for _, p in sorted(zip(s, pool, strict=True), key=lambda t: -t[0])]
                    rank = ranked.index(q["recid"]) + 1 if q["recid"] in ranked else None
                    r1.append(float(rank == 1))
                    mrr.append(1 / rank if rank and rank <= 10 else 0.0)
                name = set_name if variant == "pool" else f"{set_name}{variant if variant[0] == '+' else '@100'}"
                report[name] = {"R@1": round(statistics.fmean(r1), 3), "MRR@10": round(statistics.fmean(mrr), 3)}
    model.train()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--negatives", type=int, default=15)
    parser.add_argument("--sources", default="haiku", help="haiku, s2 or haiku,s2")
    parser.add_argument("--teacher", action="store_true", help="distil the rankings of train/teacher.py")
    parser.add_argument("--hard-weight", type=float, default=0.5, help="share of the loss on the target alone")
    parser.add_argument("--skip-head", type=int, default=0, help="negatives never from the first N candidates")
    parser.add_argument("--groups-per-step", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-only", action="store_true", help="evaluate --base and stop")
    parser.add_argument("--pools-tag", default="", help="evaluate on eval/cache/pools-*-<tag>.json")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base).to(device)

    if args.eval_only:
        print(json.dumps(evaluate(model, tok, device, eval_pools(args.pools_tag), deep=True)))
        return
    log = {"args": {k: str(v) for k, v in vars(args).items()}, "epochs": []}
    base = evaluate(model, tok, device)
    log["epochs"].append({"epoch": 0, **base})
    print("epoch 0 (base model):", json.dumps(base), flush=True)
    best = selection(base)
    args.out.mkdir(parents=True, exist_ok=True)

    teacher = json.loads((CACHE / "teacher.json").read_text()) if args.teacher else {}
    groups = training_groups(args.sources.split(","), args.negatives, args.skip_head, teacher, rng)
    print(f"{sum(g[2] is not None for g in groups)} of the groups have teacher rankings", flush=True)
    steps = max(1, args.epochs * ((len(groups) + args.groups_per_step - 1) // args.groups_per_step))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps)
    )
    print(f"{len(groups)} training groups, {steps} steps", flush=True)
    model.train()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(groups)
        start, losses = time.perf_counter(), []
        for i in range(0, len(groups), args.groups_per_step):
            chunk = groups[i : i + args.groups_per_step]
            queries, docs = [], []
            for q, group_docs, _ in chunk:
                queries += [q] * len(group_docs)
                docs += group_docs
            batch = tok(
                queries, docs, truncation="only_second", max_length=MAX_TOKENS, padding=True, return_tensors="pt"
            ).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits[:, 0].float()
            loss, start_row = torch.zeros((), device=device), 0
            for _, group_docs, probs in chunk:
                g = logits[start_row : start_row + len(group_docs)]
                start_row += len(group_docs)
                hard = -torch.log_softmax(g, dim=0)[0]  # the target is first
                if probs is None:
                    loss = loss + hard
                else:
                    target = torch.tensor(probs, device=device)
                    soft = torch.nn.functional.kl_div(torch.log_softmax(g, dim=0), target, reduction="sum")
                    loss = loss + args.hard_weight * hard + (1 - args.hard_weight) * soft
            loss = loss / len(chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        report = evaluate(model, tok, device)
        log["epochs"].append({"epoch": epoch, "loss": statistics.fmean(losses), **report})
        print(
            f"epoch {epoch}: loss {statistics.fmean(losses):.3f}, {time.perf_counter() - start:.0f}s,",
            json.dumps(report),
            flush=True,
        )
        if selection(report) > best:
            best = selection(report)
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
            log["best_epoch"] = epoch
    (args.out / "training_log.json").write_text(json.dumps(log, indent=1))
    print(
        f"best mean MRR@10 with the target inserted {best:.3f} at epoch {log.get('best_epoch', 0)}; model in {args.out}"
    )


if __name__ == "__main__":
    main()
