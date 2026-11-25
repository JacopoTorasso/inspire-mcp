"""Fine-tune the MS MARCO cross-encoder on physics questions, on the GPU, in the .venv-train environment.

Each training question (make_data.py) is one group: the target paper and N hard negatives drawn from the
INSPIRE pool the server would rerank for that question, scored together with a softmax cross-entropy
(listwise) loss, as rerankers are usually trained. Papers are encoded exactly as the server encodes them:
title, a full stop, the abstract, cut at 1200 characters (inspire_mcp.semantic.doc_text).

After each epoch the model reranks the frozen development pools of eval/rerank_lab.py (citedev, the
citation-context questions, and the older famous and longtail sets) and the best epoch on citedev is
kept. The test sets are never read here.

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
MAX_CHARS = 1200  # inspire_mcp.semantic.doc_text
MAX_TOKENS = 320


def doc_text(p: dict) -> str:
    return f"{p['title']}. {p.get('abstract') or ''}"[:MAX_CHARS]


def training_groups(negatives: int, rng: random.Random) -> list[tuple[str, str, list[str]]]:
    papers = {p["recid"]: p for p in map(json.loads, (CACHE / "papers.jsonl").read_text().splitlines())}
    questions = {q["id"]: q for q in map(json.loads, (CACHE / "questions.jsonl").read_text().splitlines())}
    groups = []
    for line in (CACHE / "pools.jsonl").read_text().splitlines():
        row = json.loads(line)
        pool = [p for p in row["pool"] if p["recid"] != row["recid"]]
        if len(pool) < negatives:
            continue
        # half from the head of the pool (what the reranker must beat), half from the rest
        head, tail = pool[:10], pool[10:]
        negs = rng.sample(head, min(len(head), negatives // 2))
        negs += rng.sample(tail, min(len(tail), negatives - len(negs)))
        groups.append((questions[row["id"]]["query"], doc_text(papers[row["recid"]]), [doc_text(p) for p in negs]))
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


def evaluate(model, tok, device: str, pool_size: int = 50) -> dict[str, dict[str, float]]:
    model.eval()
    report = {}
    for path in EVAL_POOLS:
        for set_name, queries in json.loads(path.read_text()).items():
            r1, mrr = [], []
            for q in queries:
                pool = q["pool"][:pool_size]
                s = scores(model, tok, q["query"], [doc_text(p) for p in pool], device)
                ranked = [p["recid"] for _, p in sorted(zip(s, pool, strict=True), key=lambda t: -t[0])]
                rank = ranked.index(q["recid"]) + 1 if q["recid"] in ranked else None
                r1.append(float(rank == 1))
                mrr.append(1 / rank if rank and rank <= 10 else 0.0)
            report[set_name] = {"R@1": statistics.fmean(r1), "MRR@10": statistics.fmean(mrr)}
    model.train()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--negatives", type=int, default=15)
    parser.add_argument("--groups-per-step", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base).to(device)

    log = {"args": {k: str(v) for k, v in vars(args).items()}, "epochs": []}
    base = evaluate(model, tok, device)
    log["epochs"].append({"epoch": 0, **base})
    print("epoch 0 (base model):", json.dumps(base), flush=True)
    best = base["citedev"]["MRR@10"]
    args.out.mkdir(parents=True, exist_ok=True)

    groups = training_groups(args.negatives, rng)
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
            for q, pos, negs in chunk:
                queries += [q] * (1 + len(negs))
                docs += [pos, *negs]
            batch = tok(
                queries, docs, truncation="only_second", max_length=MAX_TOKENS, padding=True, return_tensors="pt"
            ).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits[:, 0].float()
            per_group = logits.view(len(chunk), -1)  # the positive is column 0
            loss = torch.nn.functional.cross_entropy(
                per_group, torch.zeros(len(chunk), dtype=torch.long, device=device)
            )
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
        if report["citedev"]["MRR@10"] > best:
            best = report["citedev"]["MRR@10"]
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
            log["best_epoch"] = epoch
    (args.out / "training_log.json").write_text(json.dumps(log, indent=1))
    print(f"best citedev MRR@10 {best:.3f} at epoch {log.get('best_epoch', 0)}; model in {args.out}")


if __name__ == "__main__":
    main()
