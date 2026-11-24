"""Known-item questions from citation contexts: the sentence in which another paper cites the target.

A citing author describes the target from memory, in their own words, so these sentences are closer to
a real user's question than questions written from the abstract (see Limits in docs/evaluation.md).

For each sampled target (same population as the longtail set, new seed, development papers excluded):
1. a few citing papers with an arXiv id and no author in common with the target (no self-citations);
2. their LaTeX source from arXiv, read in memory (never extracted to disk), one download every 3 s;
3. citation keys of the target: its INSPIRE texkeys, plus any \\bibitem whose entry names its arXiv id;
4. the sentence around each \\cite of those keys, kept only when it cites nothing else, names no author
   of the target, repeats no three consecutive title words and has at least 8 content words.

Sentences come from arXiv papers under various licences, so only this script is versioned: the output and
the downloaded sources stay in eval/cache/ (git-ignored).

    uv run python eval/citation_contexts.py --n 10 --per-target 3
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import json
import random
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from run_eval import title_overlap  # noqa: E402
from sample_longtail import MAX_PAGE, PAGE_SIZE, SAMPLE_QUERY  # noqa: E402

from inspire_mcp.client import USER_AGENT, InspireClient  # noqa: E402
from inspire_mcp.semantic import _tokens  # noqa: E402

CACHE = HERE / "cache"
SOURCES = CACHE / "arxiv-src"
OUT = CACHE / "citation_contexts.jsonl"
EXCLUDE = ("queries.jsonl", "queries_longtail.jsonl", "queries_heldout.jsonl")
ARXIV_DELAY = 3.0
MAX_SOURCE_BYTES = 40_000_000
MIN_CONTENT_WORDS = 8
MAX_WORDS = 60

CITE = re.compile(r"\\(?:[a-zA-Z]*cite[a-zA-Z]*)\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}")
BIBITEM = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}(.*?)(?=\\bibitem|\\end\{thebibliography\}|\Z)", re.S)
COMMENT = re.compile(r"(?<!\\)%.*")
MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]", re.S)
ENVS = re.compile(r"\\begin\{(equation|align|eqnarray|figure|table|multline|gather)\*?\}.*?\\end\{\1\*?\}", re.S)
COMMANDS = re.compile(r"\\(?:ref|eqref|label|footnote|url|href)\*?\{[^}]*\}")
MARKUP = re.compile(r"\\(?:emph|textit|textbf|textrm|mathrm|it|bf|em)\b\s*")
SENTENCE_END = re.compile(r"(?<=[a-z0-9\)\]])\.\s+(?=[A-Z\\])|\n\s*\n|\\(?:sub)*section\*?\{[^}]*\}|\\item\b")


# ---------- INSPIRE ----------


async def sample_targets(client: InspireClient, n: int, seed: int, excluded: set[str]) -> list[dict]:
    rng = random.Random(seed)
    picked: dict[int, dict] = {}
    while len(picked) < n:
        data = await client._get(
            "/literature",
            {
                "q": SAMPLE_QUERY,
                "size": PAGE_SIZE,
                "page": rng.randint(1, MAX_PAGE),
                "fields": "control_number,titles.title,abstracts.value,arxiv_eprints.value,texkeys,"
                "authors.last_name,author_count",
            },
        )
        ok = []
        for h in data["hits"]["hits"]:
            m = h["metadata"]
            arxiv = (m.get("arxiv_eprints") or [{}])[0].get("value")
            abstract = (m.get("abstracts") or [{}])[0].get("value", "")
            if arxiv and arxiv not in excluded and len(abstract) >= 400 and m["control_number"] not in picked:
                ok.append(m)
        if ok:
            m = rng.choice(ok)
            picked[m["control_number"]] = {
                "recid": m["control_number"],
                "arxiv": m["arxiv_eprints"][0]["value"],
                "title": m["titles"][0]["title"],
                "abstract": m["abstracts"][0]["value"],
                "texkeys": m.get("texkeys", []),
                "authors": sorted({a["last_name"] for a in m.get("authors", []) if a.get("last_name")}),
            }
    return list(picked.values())


async def citing_papers(client: InspireClient, target: dict, k: int, rng: random.Random) -> list[str]:
    data = await client._get(
        "/literature",
        {
            "q": f"refersto:recid:{target['recid']}",
            "size": 50,
            "fields": "control_number,arxiv_eprints.value,authors.last_name",
        },
    )
    authors = {a.lower() for a in target["authors"]}
    ids = []
    for h in data["hits"]["hits"]:
        m = h["metadata"]
        arxiv = (m.get("arxiv_eprints") or [{}])[0].get("value")
        theirs = {a.get("last_name", "").lower() for a in m.get("authors", [])}
        if arxiv and not (authors & theirs):
            ids.append(arxiv)
    rng.shuffle(ids)
    return ids[: k * 2]  # some have no LaTeX source or no usable sentence


# ---------- arXiv source ----------


async def fetch_source(http: httpx.AsyncClient, arxiv: str, last: list[float]) -> bytes | None:
    path = SOURCES / (arxiv.replace("/", "_") + ".bin")
    if path.exists():
        return path.read_bytes() or None
    wait = ARXIV_DELAY - (time.monotonic() - last[0])
    if wait > 0:
        await asyncio.sleep(wait)
    last[0] = time.monotonic()
    resp = await http.get(f"https://arxiv.org/e-print/{arxiv}")
    data = resp.content if resp.status_code == 200 and len(resp.content) <= MAX_SOURCE_BYTES else b""
    SOURCES.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data or None


def latex_texts(blob: bytes) -> list[str]:
    """The .tex and .bbl files of an arXiv source package, read in memory."""
    if blob[:4] == b"%PDF":
        return []
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tar:
            out = []
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith((".tex", ".bbl")) and member.size < 5_000_000:
                    f = tar.extractfile(member)
                    if f:
                        out.append(f.read().decode("utf-8", errors="ignore"))
            return out
    except tarfile.TarError:
        pass
    try:
        return [gzip.decompress(blob).decode("utf-8", errors="ignore")]
    except OSError:
        return [blob.decode("utf-8", errors="ignore")]


# ---------- sentences ----------


def target_keys(texts: list[str], target: dict) -> set[str]:
    keys = set(target["texkeys"])
    bare = target["arxiv"]
    for text in texts:
        for key, body in BIBITEM.findall(text):
            if bare in body or any(k in body for k in target["texkeys"]):
                keys.add(key.strip())
    return keys


def clean(sentence: str) -> str:
    s = MATH.sub(" ", sentence)
    s = COMMANDS.sub(" ", s)
    s = MARKUP.sub("", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}~]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:")
    return s


def contexts(texts: list[str], keys: set[str], target: dict, stats: Counter) -> list[str]:
    found = []
    surnames = {a.lower() for a in target["authors"] if len(a) > 2}
    for text in texts:
        if "\\bibitem" in text and "\\begin{document}" not in text:
            continue  # a .bbl file: bibliography, no prose
        body = ENVS.sub(" ", COMMENT.sub("", text))
        bounds = [0, *(m.end() for m in SENTENCE_END.finditer(body)), len(body)]
        for match in CITE.finditer(body):
            cited = [k.strip() for k in match.group(1).split(",")]
            if not keys & set(cited):
                continue
            stats["cite of target"] += 1
            if len(cited) > 1:
                stats["dropped: several keys in the \\cite"] += 1
                continue
            start = max(b for b in bounds if b <= match.start())
            end = min(b for b in bounds if b > match.start())
            raw = body[start:end]
            if len(CITE.findall(raw)) > 1:
                stats["dropped: other citations in the sentence"] += 1
                continue
            sentence = clean(CITE.sub(" ", raw))
            words = re.findall(r"[a-z]+", sentence.lower())
            if surnames & set(words) or "et al" in sentence.lower():
                stats["dropped: names an author"] += 1
                continue
            if title_overlap(sentence, target["title"]):
                stats["dropped: three title words"] += 1
                continue
            if len(_tokens(sentence)) < MIN_CONTENT_WORDS or len(sentence.split()) > MAX_WORDS:
                stats["dropped: too short or too long"] += 1
                continue
            stats["kept"] += 1
            found.append(sentence)
    return found


def abstract_share(sentence: str, abstract: str) -> float:
    words = set(_tokens(sentence))
    return len(words & set(_tokens(abstract))) / len(words) if words else 0.0


# ---------- main ----------


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of target papers")
    parser.add_argument("--per-target", type=int, default=3, help="sentences kept per target, at most")
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args()
    excluded = {json.loads(line)["arxiv"] for f in EXCLUDE for line in (HERE / f).read_text().splitlines() if line}
    rng = random.Random(args.seed)
    stats: Counter = Counter()
    rows = []
    last = [0.0]
    async with (
        InspireClient() as client,
        httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as http,
    ):
        targets = await sample_targets(client, args.n, args.seed, excluded)
        for target in targets:
            kept: list[tuple[str, str]] = []
            for citing in await citing_papers(client, target, args.per_target, rng):
                if len(kept) >= args.per_target:
                    break
                stats["citing papers tried"] += 1
                blob = await fetch_source(http, citing, last)
                texts = latex_texts(blob) if blob else []
                if not texts:
                    stats["no LaTeX source"] += 1
                    continue
                before = stats["cite of target"]
                sentences = contexts(texts, target_keys(texts, target), target, stats)
                stats["citing papers where the target key was found"] += stats["cite of target"] > before
                if sentences:
                    kept.append((citing, rng.choice(sentences)))  # one sentence per citing paper
            stats["targets with a sentence"] += bool(kept)
            for i, (citing, sentence) in enumerate(kept):
                rows.append(
                    {
                        "id": f"c{target['recid']}-{i}",
                        "arxiv": target["arxiv"],
                        "recid": target["recid"],
                        "title": target["title"],
                        "query": sentence,
                        "citing_arxiv": citing,
                        "abstract_share": round(abstract_share(sentence, target["abstract"]), 3),
                    }
                )
    CACHE.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")
    if rows:
        shares = sorted(r["abstract_share"] for r in rows)
        print(f"sentences: {len(rows)} for {args.n} targets, written to {OUT.relative_to(HERE.parent)}")
        print(f"share of words found in the target abstract: median {shares[len(shares) // 2]:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
