"""Records from the other INSPIRE collections: conferences, seminars, jobs, institutions, experiments.

Each parser keeps the fields a researcher needs to act (dates, place, links) and drops the rest.
Personal contact details of job postings are left out on purpose: the posting link carries them.
"""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass
from typing import Any

TAG = re.compile(r"<[^>]+>")


def _text(value: str | None, limit: int = 600) -> str | None:
    if not value:
        return None
    clean = re.sub(r"\s+", " ", html.unescape(TAG.sub(" ", value))).strip()
    return clean if len(clean) <= limit else clean[: limit - 1].rsplit(" ", 1)[0] + "…"


def _first_url(meta: dict[str, Any]) -> str | None:
    urls = meta.get("urls") or []
    return urls[0].get("value") if urls else None


def _place(addresses: list[dict[str, Any]] | None, limit: int = 2) -> str | None:
    places = []
    for a in (addresses or [])[:limit]:
        parts = [a.get("place_name"), ", ".join(a.get("cities") or []) or None, a.get("country")]
        if label := ", ".join(p for p in parts if p):
            places.append(label)
    return "; ".join(places) or None


@dataclass
class Record:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Conference(Record):
    recid: int
    title: str
    acronym: str | None
    opening_date: str | None
    closing_date: str | None
    place: str | None
    website: str | None
    categories: list[str]
    url: str


@dataclass
class Seminar(Record):
    recid: int
    title: str
    start: str | None
    end: str | None
    timezone: str | None
    speakers: list[str]
    series: str | None
    join_url: str | None
    url: str


@dataclass
class Job(Record):
    recid: int
    position: str
    institutions: list[str]
    regions: list[str]
    ranks: list[str]
    fields: list[str]
    deadline: str | None
    status: str | None
    description: str | None
    apply_url: str | None
    url: str


@dataclass
class Institution(Record):
    recid: int
    name: str
    place: str | None
    papers: int | None
    website: str | None
    url: str


@dataclass
class Experiment(Record):
    recid: int
    name: str
    long_name: str | None
    accelerator: str | None
    approved: str | None
    started: str | None
    completed: str | None
    papers: int | None
    website: str | None
    url: str


def parse_conference(m: dict[str, Any]) -> Conference:
    recid = int(m["control_number"])
    return Conference(
        recid=recid,
        title=(m.get("titles") or [{}])[0].get("title", ""),
        acronym=(m.get("acronyms") or [None])[0],
        opening_date=m.get("opening_date"),
        closing_date=m.get("closing_date"),
        place=_place(m.get("addresses")),
        website=_first_url(m),
        categories=[c["term"] for c in m.get("inspire_categories", []) if "term" in c],
        url=f"https://inspirehep.net/conferences/{recid}",
    )


def parse_seminar(m: dict[str, Any]) -> Seminar:
    recid = int(m["control_number"])
    speakers = []
    for s in m.get("speakers", []):
        affiliation = ", ".join(a.get("value", "") for a in s.get("affiliations", []) if a.get("value"))
        speakers.append(f"{s.get('name', '')} ({affiliation})" if affiliation else s.get("name", ""))
    join = m.get("join_urls") or []
    series = m.get("series") or []
    return Seminar(
        recid=recid,
        title=(m.get("title") or {}).get("title", ""),
        start=m.get("start_datetime"),
        end=m.get("end_datetime"),
        timezone=m.get("timezone"),
        speakers=speakers,
        series=series[0].get("name") if series else None,
        join_url=join[0].get("value") if join else None,
        url=f"https://inspirehep.net/seminars/{recid}",
    )


def parse_job(m: dict[str, Any]) -> Job:
    recid = int(m["control_number"])
    return Job(
        recid=recid,
        position=m.get("position", ""),
        institutions=[i.get("value", "") for i in m.get("institutions", []) if i.get("value")],
        regions=m.get("regions", []),
        ranks=m.get("ranks", []),
        fields=m.get("arxiv_categories", []),
        deadline=m.get("deadline_date"),
        status=m.get("status"),
        description=_text(m.get("description")),
        apply_url=_first_url(m),
        url=f"https://inspirehep.net/jobs/{recid}",
    )


def parse_institution(m: dict[str, Any]) -> Institution:
    recid = int(m["control_number"])
    names = m.get("legacy_ICN") or (m.get("ICN") or [None])[0] or ""
    return Institution(
        recid=recid,
        name=names if isinstance(names, str) else str(names),
        place=_place(m.get("addresses")),
        papers=m.get("number_of_papers"),
        website=_first_url(m),
        url=f"https://inspirehep.net/institutions/{recid}",
    )


def parse_experiment(m: dict[str, Any]) -> Experiment:
    recid = int(m["control_number"])
    return Experiment(
        recid=recid,
        name=m.get("legacy_name", ""),
        long_name=m.get("long_name"),
        accelerator=(m.get("accelerator") or {}).get("value"),
        approved=m.get("date_approved"),
        started=m.get("date_started"),
        completed=m.get("date_completed"),
        papers=m.get("number_of_papers"),
        website=_first_url(m),
        url=f"https://inspirehep.net/experiments/{recid}",
    )
