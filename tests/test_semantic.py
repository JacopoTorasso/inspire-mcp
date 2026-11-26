import httpx
import pytest
import respx

from inspire_mcp.client import BASE_URL, InspireClient, Paper
from inspire_mcp.semantic import (
    BM25Reranker,
    candidates,
    content_terms,
    interleave,
    phrase_terms,
    relaxed_query,
    rrf,
    semantic_search,
)

from .conftest import ATLAS_HIGGS, hits


def paper(recid: int, title: str, abstract: str = "") -> Paper:
    return Paper(recid, title, abstract, [], None, [], None, None, None, None, None)


def test_content_terms_drop_stopwords_reserved_words_and_duplicates():
    q = "What is the paper that found the Higgs and the Higgs mass at 125 GeV, version 8.2?"
    assert content_terms(q) == ["higgs", "mass", "125", "gev", "version", "8.2"]


def test_relaxed_query_quotes_hyphenated_and_dotted_terms():
    assert (
        relaxed_query("D-branes carry RR charge, PYTHIA 8.2")
        == '"d-branes" or carry or rr or charge or pythia or "8.2"'
    )


@respx.mock
async def test_candidates_survive_invalid_syntax():
    route = respx.get(f"{BASE_URL}/literature").mock(
        side_effect=[httpx.Response(400), httpx.Response(400), httpx.Response(200, json=hits(ATLAS_HIGGS))]
    )
    async with InspireClient(min_interval=0) as client:
        docs, relaxed, strict_total = await candidates(client, "weird query", pool=10)
    assert strict_total == 0 and [d.recid for d in docs] == [1124337]
    assert relaxed == '"weird" or "query"' and route.call_count == 3


def test_relaxed_query_joins_with_or():
    assert relaxed_query("warped extra dimension") == "warped or extra or dimension"
    assert relaxed_query("what is the") == ""


def test_bm25_prefers_matching_document():
    docs = [paper(1, "Jet clustering algorithm"), paper(2, "Warped extra dimension", "hierarchy problem")]
    scores = BM25Reranker().scores("warped dimension hierarchy", docs)
    assert scores[1] > scores[0]


def test_rrf_rewards_agreement():
    assert rrf([1, 2, 3], [2, 1, 3])[:2] in ([1, 2], [2, 1])
    assert rrf([1, 2], [3, 2])[0] == 2  # the only record both rankers retrieved


class FixedReranker:
    name = "fixed"
    fuse_with_inspire = False

    def __init__(self, best: int) -> None:
        self.best = best

    def scores(self, query, docs):
        return [1.0 if d.recid == self.best else 0.0 for d in docs]


@respx.mock
async def test_candidates_merge_strict_then_relaxed_without_duplicates():
    other = {**ATLAS_HIGGS, "control_number": 2}
    route = respx.get(f"{BASE_URL}/literature").mock(
        side_effect=[httpx.Response(200, json=hits(ATLAS_HIGGS)), httpx.Response(200, json=hits(ATLAS_HIGGS, other))]
    )
    async with InspireClient(min_interval=0) as client:
        docs, relaxed, strict_total = await candidates(client, "higgs boson discovery", pool=10)
    assert [d.recid for d in docs] == [1124337, 2]
    assert relaxed == "higgs or boson or discovery" and strict_total == 1
    assert route.calls[1].request.url.params["q"] == "higgs or boson or discovery"


@respx.mock
async def test_semantic_search_reranks_with_reranker_and_reports_retrieval():
    a, b = {**ATLAS_HIGGS, "control_number": 10}, {**ATLAS_HIGGS, "control_number": 20}
    respx.get(f"{BASE_URL}/literature").mock(
        side_effect=[httpx.Response(200, json=hits(total=0)), httpx.Response(200, json=hits(a, b))]
    )
    async with InspireClient(min_interval=0) as client:
        result = await semantic_search(client, "higgs", k=2, reranker=FixedReranker(20), fuse_with_inspire=False)
    assert [p.recid for p in result.papers] == [20, 10]
    assert result.strict_hits == 0 and result.pool_size == 2 and result.reranker == "fixed"


class FakeEmbedding:
    """Stands in for fastembed.TextEmbedding: bag-of-letters vectors, counts how many texts it embeds."""

    embedded = 0

    def __init__(self, model):
        self.model = model

    def embed(self, texts, batch_size=32):
        import numpy as np

        for t in texts:
            FakeEmbedding.embedded += 1
            v = np.zeros(26, dtype=np.float32)
            for ch in t.lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1
            yield v + 1e-3


def test_dense_reranker_scores_and_caches_vectors(tmp_path, monkeypatch):
    import sys
    import types

    from inspire_mcp.semantic import DenseReranker

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeEmbedding))
    FakeEmbedding.embedded = 0
    docs = [paper(1, "zzzz zzzz"), paper(2, "holographic entanglement entropy")]
    reranker = DenseReranker(cache_path=tmp_path)
    scores = reranker.scores("entanglement entropy", docs)
    assert scores[1] > scores[0]
    first = FakeEmbedding.embedded  # 2 documents + 1 query
    reranker.scores("entanglement entropy", docs)
    assert FakeEmbedding.embedded - first == 1  # only the query is embedded again
    assert (tmp_path / "embeddings.sqlite").exists()
    assert reranker.name == "dense:bge-small-en-v1.5"


def test_phrase_terms_pairs_adjacent_content_words_only():
    assert phrase_terms("the hierarchy problem and new warped extra dimensions") == [
        "hierarchy problem",
        "warped extra",
        "extra dimensions",
    ]


def test_interleave_round_robin_without_duplicates():
    a, b, c = paper(1, "a"), paper(2, "b"), paper(3, "c")
    assert [p.recid for p in interleave([a, b], [b, c], [c])] == [1, 2, 3]


@respx.mock
async def test_multi_strategy_queries_phrases_and_titles():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with InspireClient(min_interval=0) as client:
        await candidates(client, "warped extra dimension", pool=10, strategy="multi")
    queries = [c.request.url.params["q"] for c in route.calls]
    assert '"warped extra" or "extra dimension"' in queries
    assert "t warped or t extra or t dimension" in queries


@respx.mock
async def test_fusion_follows_the_reranker_preference():
    a, b = {**ATLAS_HIGGS, "control_number": 10}, {**ATLAS_HIGGS, "control_number": 20}
    respx.get(f"{BASE_URL}/literature").mock(
        side_effect=[httpx.Response(200, json=hits(total=0)), httpx.Response(200, json=hits(a, b))] * 2
    )
    async with InspireClient(min_interval=0, cache_ttl=0) as client:
        plain = await semantic_search(client, "higgs", k=2, reranker=FixedReranker(20))
        bm25 = await semantic_search(client, "higgs", k=2, reranker=BM25Reranker())
    assert plain.fusion == "reranker only" and [p.recid for p in plain.papers] == [20, 10]
    assert bm25.fusion == "rrf(reranker, inspire)"


def test_default_reranker_falls_back_when_models_are_missing(monkeypatch):
    import inspire_mcp.semantic as sem

    def unavailable(*args, **kwargs):
        raise RuntimeError("no model")

    monkeypatch.setattr(sem, "CrossEncoderReranker", unavailable)
    monkeypatch.setattr(sem, "DenseReranker", unavailable)
    assert isinstance(sem.default_reranker(), BM25Reranker)


class FakeCrossEncoder:
    registered: list[str] = []
    loaded: list[tuple[str, str | None]] = []

    @classmethod
    def list_supported_models(cls):
        return [{"model": "Xenova/ms-marco-MiniLM-L-6-v2"}, *({"model": m} for m in cls.registered)]

    @classmethod
    def add_custom_model(cls, name, sources, model_file):
        if name in cls.registered:
            raise ValueError("already registered")
        cls.registered.append(name)

    def __init__(self, name, specific_model_path=None):
        self.loaded.append((name, specific_model_path))


def test_cross_encoder_from_fastembed_local_directory_or_hub(tmp_path):
    pytest.importorskip("fastembed")
    from inspire_mcp.semantic import load_cross_encoder

    FakeCrossEncoder.registered, FakeCrossEncoder.loaded = [], []
    load_cross_encoder(FakeCrossEncoder, "Xenova/ms-marco-MiniLM-L-6-v2")
    model_dir = tmp_path / "physics-reranker"
    model_dir.mkdir()
    load_cross_encoder(FakeCrossEncoder, str(model_dir))
    load_cross_encoder(FakeCrossEncoder, str(model_dir))  # a second load must not register the name again
    load_cross_encoder(FakeCrossEncoder, "someone/physics-reranker-onnx")
    assert FakeCrossEncoder.loaded == [
        ("Xenova/ms-marco-MiniLM-L-6-v2", None),
        ("local/physics-reranker", str(model_dir)),
        ("local/physics-reranker", str(model_dir)),
        ("someone/physics-reranker-onnx", None),
    ]
    assert FakeCrossEncoder.registered == ["local/physics-reranker", "someone/physics-reranker-onnx"]


class CountingReranker:
    name = "counting"
    fuse_with_inspire = False
    prefilter = 2

    def __init__(self):
        self.seen = []

    def scores(self, query, docs):
        self.seen.append([d.recid for d in docs])
        return [float(d.recid) for d in docs]


@respx.mock
async def test_prefilter_scores_only_the_bm25_head_and_keeps_the_rest():
    docs = [
        {**ATLAS_HIGGS, "control_number": 1, "titles": [{"title": "Unrelated cosmology"}], "abstracts": []},
        {**ATLAS_HIGGS, "control_number": 2},
        {**ATLAS_HIGGS, "control_number": 3},
    ]
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(*docs)))
    reranker = CountingReranker()
    async with InspireClient(min_interval=0, cache_ttl=0) as client:
        result = await semantic_search(client, "higgs boson discovery", k=3, reranker=reranker)
    assert sorted(reranker.seen[0]) == [2, 3]  # the paper with no matching word is not scored
    assert [p.recid for p in result.papers] == [3, 2, 1]
    assert set(result.scores) == {2, 3}
