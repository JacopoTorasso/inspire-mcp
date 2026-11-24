ATLAS_HIGGS = {
    "control_number": 1124337,
    "titles": [{"title": "Observation of a new particle in the search for the Standard Model Higgs boson"}],
    "abstracts": [
        {"source": "Elsevier", "value": "publisher abstract with mangled math"},
        {"source": "arXiv", "value": "A search for the Standard Model Higgs boson in proton-proton collisions."},
    ],
    "authors": [{"full_name": f"Author, {i}"} for i in range(8)],
    "author_count": 2932,
    "collaborations": [{"value": "ATLAS"}],
    "arxiv_eprints": [{"value": "1207.7214", "categories": ["hep-ex"]}],
    "dois": [{"value": "10.1016/j.physletb.2012.08.020"}],
    "publication_info": [{"journal_title": "Phys.Lett.B", "journal_volume": "716", "page_start": "1", "year": 2012}],
    "citation_count": 17848,
    "inspire_categories": [{"term": "Experiment-HEP"}],
}

MATHML_TITLE = {
    "control_number": 2678048,
    "titles": [{"title": 'Observation of a <math display="inline"><mi>J</mi><mo>/</mo><mi>ψ</mi></math> state'}],
    "earliest_date": "2023-04-01",
}

WITTEN = {
    "control_number": 983328,
    "name": {"value": "Witten, Edward", "preferred_name": "Edward Witten"},
    "ids": [{"schema": "INSPIRE BAI", "value": "Edward.Witten.1"}, {"schema": "ORCID", "value": "0000-0002-7752-6073"}],
    "positions": [{"institution": "Princeton, Inst. Advanced Study", "current": True}, {"institution": "Princeton U."}],
    "arxiv_categories": ["hep-th"],
}


def hits(*records, total=None):
    return {"hits": {"total": total if total is not None else len(records), "hits": [{"metadata": r} for r in records]}}
