"""Async client for the public INSPIRE-HEP REST API.

Docs: https://github.com/inspirehep/rest-api-doc
The API is public and needs no key, but it is rate limited, so requests are
spaced out and 429 responses are retried after the reset time the server returns.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import httpx

BASE_URL = "https://inspirehep.net/api"
USER_AGENT = "inspire-mcp/0.1 (+https://github.com/inspirehep/rest-api-doc)"

LITERATURE_FIELDS = ",".join(
    [
        "control_number",
        "titles.title",
        "abstracts",
        "authors.full_name",
        "author_count",
        "collaborations.value",
        "arxiv_eprints",
        "dois.value",
        "publication_info",
        "citation_count",
        "earliest_date",
        "inspire_categories.term",
        "keywords.value",
    ]
)
AUTHOR_FIELDS = "control_number,name,ids,positions,arxiv_categories"

Sort = Literal["mostrecent", "mostcited"]

ARXIV_NEW = re.compile(r"^(arxiv:)?\d{4}\.\d{4,5}(v\d+)?$", re.IGNORECASE)
TAG = re.compile(r"<[^>]+>")
ARXIV_OLD = re.compile(r"^(arxiv:)?[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$", re.IGNORECASE)


class InspireError(RuntimeError):
    pass


@dataclass
class Paper:
    recid: int
    title: str
    abstract: str | None
    authors: list[str]
    author_count: int | None
    collaborations: list[str]
    year: int | None
    journal: str | None
    arxiv_id: str | None
    doi: str | None
    citation_count: int | None
    categories: list[str] = field(default_factory=list)
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Author:
    recid: int
    name: str
    bai: str | None
    orcid: str | None
    current_institution: str | None
    arxiv_categories: list[str]
    url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_markup(text: str) -> str:
    """Drop MathML/HTML tags some publisher titles carry, keeping their text content."""
    return re.sub(r"\s+", " ", TAG.sub("", text)).strip() if "<" in text else text


def _pick_abstract(abstracts: list[dict[str, Any]] | None) -> str | None:
    """Prefer the arXiv abstract: publisher abstracts often contain mangled math."""
    if not abstracts:
        return None
    for a in abstracts:
        if a.get("source") == "arXiv":
            return a.get("value")
    return abstracts[0].get("value")


def parse_paper(metadata: dict[str, Any], max_authors: int = 5) -> Paper:
    recid = int(metadata["control_number"])
    titles = metadata.get("titles") or [{}]
    authors = [a["full_name"] for a in metadata.get("authors", [])[:max_authors] if "full_name" in a]
    pub = (metadata.get("publication_info") or [{}])[0]
    journal = None
    if pub.get("journal_title"):
        journal = " ".join(
            str(p)
            for p in (pub.get("journal_title"), pub.get("journal_volume"), pub.get("artid") or pub.get("page_start"))
            if p
        )
    year = pub.get("year")
    if year is None and metadata.get("earliest_date"):
        year = int(str(metadata["earliest_date"])[:4])
    eprints = metadata.get("arxiv_eprints") or []
    dois = metadata.get("dois") or []
    return Paper(
        recid=recid,
        title=clean_markup(titles[0].get("title", "")),
        abstract=_pick_abstract(metadata.get("abstracts")),
        authors=authors,
        author_count=metadata.get("author_count", len(metadata.get("authors", [])) or None),
        collaborations=[c["value"] for c in metadata.get("collaborations", []) if "value" in c],
        year=year,
        journal=journal,
        arxiv_id=eprints[0].get("value") if eprints else None,
        doi=dois[0].get("value") if dois else None,
        citation_count=metadata.get("citation_count"),
        categories=[c["term"] for c in metadata.get("inspire_categories", []) if "term" in c],
        url=f"https://inspirehep.net/literature/{recid}",
    )


def parse_author(metadata: dict[str, Any]) -> Author:
    recid = int(metadata["control_number"])
    ids = {i.get("schema"): i.get("value") for i in metadata.get("ids", [])}
    name = metadata.get("name", {})
    current = next((p.get("institution") for p in metadata.get("positions", []) if p.get("current")), None)
    return Author(
        recid=recid,
        name=name.get("preferred_name") or name.get("value", ""),
        bai=ids.get("INSPIRE BAI"),
        orcid=ids.get("ORCID"),
        current_institution=current,
        arxiv_categories=metadata.get("arxiv_categories", []),
        url=f"https://inspirehep.net/authors/{recid}",
    )


def identifier_query(identifier: str) -> str:
    """Translate a recid, arXiv id or DOI into an INSPIRE search query."""
    ident = identifier.strip()
    if ident.isdigit():
        return f"recid:{ident}"
    if ident.lower().startswith("doi:"):
        ident = ident[4:]
    if ident.startswith("10."):
        return f'doi:"{ident}"'
    if ARXIV_NEW.match(ident) or ARXIV_OLD.match(ident):
        ident = re.sub(r"^arxiv:", "", ident, flags=re.IGNORECASE)
        return f"arxiv:{re.sub(r'v\d+$', '', ident)}"
    raise InspireError(f"Unrecognised identifier {identifier!r}: expected an INSPIRE recid, arXiv id or DOI")


class InspireClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        min_interval: float = 0.35,
        max_retries: int = 3,
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
        cache_ttl: float = 300.0,
        cache_size: int = 512,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._lock = asyncio.Lock()
        self._last_request = 0.0
        self._cache_ttl = cache_ttl
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> InspireClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _request(self, path: str, params: dict[str, Any], as_text: bool = False) -> Any:
        key = f"{path}?{sorted(params.items())}|{as_text}"
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self._cache_ttl:
            self._cache.move_to_end(key)
            return hit[1]
        for attempt in range(self._max_retries + 1):
            async with self._lock:
                wait = self._min_interval - (time.monotonic() - self._last_request)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_request = time.monotonic()
                try:
                    headers = {"Accept": "application/x-bibtex"} if params.get("format") == "bibtex" else None
                    resp = await self._http.get(path, params=params, headers=headers)
                except httpx.TransportError as exc:
                    if attempt < self._max_retries:
                        continue
                    raise InspireError(f"INSPIRE API unreachable: {exc.__class__.__name__}") from exc
            if resp.status_code in (429, 502, 503, 504) and attempt < self._max_retries:
                await asyncio.sleep(_retry_after(resp, attempt))
                continue
            if resp.status_code >= 400:
                raise InspireError(f"INSPIRE API returned HTTP {resp.status_code} for {path}")
            value = resp.text if as_text else resp.json()
            self._cache[key] = (time.monotonic(), value)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return value
        raise InspireError("INSPIRE API unavailable: retries exhausted")

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._request(path, params)

    async def search_literature(
        self, query: str, size: int = 10, sort: Sort | None = None, page: int = 1
    ) -> tuple[int, list[Paper]]:
        params: dict[str, Any] = {"q": query, "size": size, "page": page, "fields": LITERATURE_FIELDS}
        if sort:
            params["sort"] = sort
        data = await self._get("/literature", params)
        hits = data["hits"]
        return hits["total"], [parse_paper(h["metadata"]) for h in hits["hits"]]

    async def get_paper(self, identifier: str) -> Paper | None:
        _, papers = await self.search_literature(identifier_query(identifier), size=1)
        return papers[0] if papers else None

    async def citing_papers(
        self, recid: int, size: int = 10, sort: Sort | None = "mostrecent"
    ) -> tuple[int, list[Paper]]:
        return await self.search_literature(f"refersto:recid:{recid}", size=size, sort=sort)

    async def references(self, recid: int, size: int = 25) -> tuple[int, list[Paper]]:
        return await self.search_literature(f"citedby:recid:{recid}", size=size, sort="mostcited")

    async def bibtex(self, identifiers: list[str]) -> str:
        """BibTeX entries, as generated by INSPIRE, for recids, arXiv ids or DOIs."""
        query = " or ".join(identifier_query(i) for i in identifiers)
        return await self._request(
            "/literature", {"q": query, "size": len(identifiers), "format": "bibtex"}, as_text=True
        )

    async def search_authors(self, name: str, size: int = 5) -> list[Author]:
        data = await self._get("/authors", {"q": name, "size": size, "fields": AUTHOR_FIELDS})
        return [parse_author(h["metadata"]) for h in data["hits"]["hits"]]


def _retry_after(resp: httpx.Response, attempt: int) -> float:
    reset = resp.headers.get("Retry-After") or resp.headers.get("X-RateLimit-Reset")
    if reset:
        try:
            value = float(reset)
            # X-RateLimit-Reset may be an absolute epoch timestamp.
            return max(0.5, min(value - time.time() if value > 1e9 else value, 30.0))
        except ValueError:
            pass
    return min(2.0**attempt, 10.0)
