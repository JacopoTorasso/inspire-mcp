from datetime import date

import httpx
import pytest
import respx

from inspire_mcp.client import BASE_URL, InspireClient
from inspire_mcp.semantic import candidates

from .conftest import ATLAS_HIGGS, hits

np = pytest.importorskip("numpy")


class FakeEmbedder:
    """Maps a text to a one-hot vector by its first word: 'alpha ...' -> e0, 'beta ...' -> e1, ..."""

    WORDS = ("alpha", "beta", "gamma", "delta")

    def __init__(self):
        self.seen = []

    def embed(self, texts, batch_size=64):
        for t in texts:
            self.seen.append(t)
            word = t.split(": ")[-1].split()[0].lower().strip(".")
            v = np.zeros(len(self.WORDS), dtype=np.float32)
            v[self.WORDS.index(word) if word in self.WORDS else 0] = 1.0
            yield v


def write_index(path, recids, vectors, prefix="q: "):
    import json

    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "index.npz", recids=np.array(recids, dtype=np.int64), vectors=np.array(vectors, dtype=np.float16))
    meta = {"model": "fake", "query_prefix": prefix, "records": len(recids), "newest": "2026-11-01"}
    (path / "index.json").write_text(json.dumps(meta))


def test_dense_index_returns_nearest_records_and_adds_the_query_prefix(tmp_path):
    from inspire_mcp.dense import DenseIndex

    eye = np.eye(4)
    write_index(tmp_path, [10, 20, 30, 40], [eye[0], eye[1], 0.6 * eye[1] + 0.8 * eye[2], eye[3]])
    embedder = FakeEmbedder()
    index = DenseIndex(tmp_path, embedder=embedder)
    assert index.search("beta decay", k=2) == [20, 30]
    assert embedder.seen == ["q: beta decay"]
    assert index.search("delta", k=10)[0] == 40  # k larger than the index


def test_dense_index_rejects_mismatched_files(tmp_path):
    from inspire_mcp.dense import DenseIndex

    write_index(tmp_path, [1, 2], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        DenseIndex(tmp_path, embedder=FakeEmbedder())


class FixedDense:
    def __init__(self, recids):
        self.recids = recids

    def search(self, question, k):
        return self.recids[:k]


@respx.mock
async def test_candidates_interleave_inspire_and_dense_and_fetch_only_unknown_records():
    a, b, c = ({**ATLAS_HIGGS, "control_number": n} for n in (1, 2, 3))
    route = respx.get(f"{BASE_URL}/literature")

    def answer(request):
        q = request.url.params["q"]
        return httpx.Response(200, json=hits(c) if q.startswith("recid:") else hits(a, b))

    route.mock(side_effect=answer)
    async with InspireClient(min_interval=0, cache_ttl=0) as client:
        docs, _, _ = await candidates(client, "higgs boson", pool=4, dense=FixedDense([3, 1]))
    assert [d.recid for d in docs] == [1, 3, 2]  # INSPIRE first, then the index, without duplicates
    fetched = [c.request.url.params["q"] for c in route.calls if c.request.url.params["q"].startswith("recid:")]
    assert fetched == ["recid:3"]  # record 1 was already among INSPIRE's results


async def test_build_then_update_replaces_records_and_keeps_one_row_each(tmp_path, monkeypatch):
    import inspire_mcp.dense as dense

    batches = [[(1, "alpha one", "2024-01-05"), (2, "beta two", "2024-02-01")], [(2, "gamma two", "2024-02-01")]]

    async def fake_harvest(client, since):
        for batch in batches:
            yield batch

    monkeypatch.setattr(dense, "harvest", fake_harvest)
    monkeypatch.setattr(dense, "_embedder", lambda model: FakeEmbedder())
    assert await dense.build(tmp_path, model="BAAI/bge-small-en-v1.5", client=InspireClient()) == 3
    index = dense.DenseIndex(tmp_path, embedder=FakeEmbedder())
    assert index.records == 2 and index.newest == "2024-02-01"
    assert index.prefix.startswith("Represent this sentence")

    batches = [[(3, "delta three", "2024-03-01"), (1, "beta one", "2024-01-05")]]
    assert await dense.build(tmp_path, update=True, client=InspireClient()) == 2
    index = dense.DenseIndex(tmp_path, embedder=FakeEmbedder())
    assert sorted(np.load(tmp_path / "index.npz")["recids"].tolist()) == [1, 2, 3] and index.newest == "2024-03-01"
    assert not list(tmp_path.glob("*.tmp"))  # written aside, then renamed
    assert index.search("beta", k=1) == [1]  # record 1 re-embedded as 'beta'


@respx.mock
async def test_harvest_splits_months_over_the_result_limit():
    import inspire_mcp.dense as dense

    def answer(request):
        q = request.url.params["q"]
        if "2024-01-01->2024-01-31" in q:  # the whole month is too many: it must be halved
            return httpx.Response(200, json=hits(total=dense.RESULT_LIMIT + 1))
        rec = {**ATLAS_HIGGS, "earliest_date": "2024-01-10"}
        return httpx.Response(200, json=hits(rec))

    respx.get(f"{BASE_URL}/literature").mock(side_effect=answer)
    monkey_today = date(2024, 1, 31)
    async with InspireClient(min_interval=0, cache_ttl=0) as client:
        months = [b async for b in _harvest_until(dense, client, date(2024, 1, 1), monkey_today)]
    assert len(months) == 1 and len(months[0]) == 2  # two halves of January, one record each
    recid, text, earliest = months[0][0]
    assert recid == ATLAS_HIGGS["control_number"] and "proton-proton" in text and earliest == "2024-01-10"


async def _harvest_until(dense, client, since, today):
    import inspire_mcp.dense as module

    real = module.date

    class FrozenDate(real):
        @classmethod
        def today(cls):
            return today

    module.date = FrozenDate
    try:
        async for batch in dense.harvest(client, since):
            yield batch
    finally:
        module.date = real


@respx.mock
async def test_server_reloads_the_dense_index_when_the_file_changes(tmp_path, monkeypatch):
    import asyncio
    import os

    from mcp import Client

    import inspire_mcp.server as srv
    from inspire_mcp.semantic import BM25Reranker

    a, b = ({**ATLAS_HIGGS, "control_number": n} for n in (1, 2))
    respx.get(f"{BASE_URL}/literature").mock(
        side_effect=lambda r: httpx.Response(200, json=hits(b) if r.url.params["q"].startswith("recid:") else hits(a))
    )
    loads = []

    class CountingIndex:
        def __init__(self, path):
            loads.append(path)
            self.name, self.newest = f"fake:{len(loads)}", None

        def search(self, question, k):
            return [2]

    monkeypatch.setattr(srv, "DenseIndex", CountingIndex)
    monkeypatch.setattr(srv, "DENSE_RELOAD_CHECK", 0.0)
    write_index(tmp_path, [2], [[1.0, 0.0]])
    server = srv.build_server(InspireClient(min_interval=0, cache_ttl=0), reranker=BM25Reranker(), dense_index=tmp_path)
    async with Client(server) as client:
        first = await client.call_tool("semantic_search", {"question": "higgs boson"})
        assert first.structured_content["retrieval"]["dense_index"] == "fake:1"
        assert {r["recid"] for r in first.structured_content["results"]} == {1, 2}
        os.utime(tmp_path / "index.npz", (1, 1))  # an update replaced the file
        await client.call_tool("semantic_search", {"question": "higgs boson"})
        for _ in range(50):
            await asyncio.sleep(0.01)
            if len(loads) == 2:
                break
        third = await client.call_tool("semantic_search", {"question": "higgs boson"})
    assert len(loads) == 2 and third.structured_content["retrieval"]["dense_index"] == "fake:2"
