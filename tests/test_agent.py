import json

from conftest import StubGenerator

from taglish_rag.agent.crag import _format_context, _heuristic_grade, build_crag_graph, run_naive_rag
from taglish_rag.agent.grader import LLMGrader, format_passages
from taglish_rag.schemas import RetrievedChunk


def stub_grader(**payload) -> LLMGrader:
    """An LLMGrader whose backend returns one canned JSON verdict."""
    return LLMGrader(StubGenerator(json.dumps(payload)))


class FakeRetriever:
    """Minimal stand-in for taglish_rag.retrieval.retriever.Retriever, so
    the CRAG graph's routing logic can be tested without loading real
    chunk files or embedding models."""

    def __init__(self, score: float, title: str = "Doc A", extra_chunks: list[dict] | None = None):
        self._score = score
        self.chunks = [
            {"chunk_id": "docA__0", "doc_id": "docA", "title": title, "text": "Doc A content about penalties."}
        ] + (extra_chunks or [])
        self.chunks_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.texts_by_id = {c["chunk_id"]: c["text"] for c in self.chunks}

    def retrieve(self, query: str):
        return [
            RetrievedChunk(chunk_id=c["chunk_id"], score=self._score, rank=i)
            for i, c in enumerate(self.chunks)
        ]


def test_heuristic_grade_strong_above_threshold():
    assert _heuristic_grade([RetrievedChunk(chunk_id="x", score=0.9, rank=0)]) == "strong"


def test_heuristic_grade_weak_below_threshold():
    assert _heuristic_grade([RetrievedChunk(chunk_id="x", score=0.01, rank=0)]) == "weak"


def test_heuristic_grade_weak_when_nothing_retrieved():
    assert _heuristic_grade([]) == "weak"


def test_crag_graph_strong_retrieval_skips_rewrite():
    retriever = FakeRetriever(score=0.9)
    app = build_crag_graph(retriever, StubGenerator())
    final_state = app.invoke({"question": "What are the penalties?", "rewrite_count": 0})
    assert final_state["grade"] == "strong"
    assert final_state.get("rewrite_count", 0) == 0
    assert "answer" in final_state


def test_crag_graph_weak_retrieval_triggers_one_rewrite_then_stops(monkeypatch):
    # Stub out the real (network-downloading) MarianMT translator so this
    # stays a fast, offline unit test of the graph's routing logic.
    import taglish_rag.agent.crag as crag_module

    monkeypatch.setattr(crag_module, "_heuristic_rewrite", lambda q: q + " (rewritten)")

    retriever = FakeRetriever(score=0.01)  # always weak, even after rewrite
    app = build_crag_graph(retriever, StubGenerator())
    final_state = app.invoke({"question": "Ano ang parusa?", "rewrite_count": 0})
    # MAX_REWRITES=1, so it should rewrite exactly once then proceed to generate
    assert final_state["rewrite_count"] == 1
    assert "answer" in final_state


def test_grader_parses_verdict_and_maps_to_strong():
    result = stub_grader(
        verdict="correct", sufficient_passages=[1], missing="", corpus_gap=False
    ).grade("What are the penalties?", ["Doc A content about penalties."])
    assert result.parse_ok
    assert result.verdict == "correct"
    assert result.sufficient_passages == [1]
    assert result.grade == "strong"


def test_grader_ambiguous_is_weak():
    result = stub_grader(verdict="ambiguous", missing="no rate for 2025").grade("q", ["p"])
    assert result.grade == "weak"
    assert result.missing == "no rate for 2025"


def test_grader_drops_out_of_range_passage_indices():
    result = stub_grader(verdict="correct", sufficient_passages=[1, 7, "x"]).grade("q", ["p"])
    assert result.sufficient_passages == [1]


def test_grader_flags_unparseable_response():
    result = LLMGrader(StubGenerator("I think the passages look fine.")).grade("q", ["p"])
    assert not result.parse_ok


def test_grader_rejects_unrecognized_verdict():
    result = stub_grader(verdict="maybe").grade("q", ["p"])
    assert not result.parse_ok


def test_grader_short_circuits_on_empty_retrieval():
    # Should not spend a backend call to conclude nothing was retrieved.
    result = LLMGrader(StubGenerator("should never be read")).grade("q", [])
    assert result.verdict == "incorrect"
    assert result.grade == "weak"


def test_format_passages_is_one_based():
    assert format_passages(["a", "b"]) == "[1] a\n\n[2] b"


def test_crag_graph_uses_grader_verdict_over_retrieval_score(monkeypatch):
    # A weak grade routes through rewrite; stub the MarianMT translator so
    # this stays offline (see the rewrite test above).
    import taglish_rag.agent.crag as crag_module

    monkeypatch.setattr(crag_module, "_heuristic_rewrite", lambda q: q + " (rewritten)")

    # High score would be "strong" heuristically; the grader says otherwise.
    retriever = FakeRetriever(score=0.9)
    app = build_crag_graph(
        retriever, StubGenerator(), stub_grader(verdict="incorrect", missing="wrong agency")
    )
    final_state = app.invoke({"question": "Magkano ang SSS contribution?", "rewrite_count": 0})
    assert final_state["verdict"] == "incorrect"
    assert final_state["grade"] == "weak"


def test_crag_graph_corpus_gap_skips_rewrite():
    retriever = FakeRetriever(score=0.01)
    app = build_crag_graph(
        retriever, StubGenerator(), stub_grader(verdict="incorrect", corpus_gap=True)
    )
    final_state = app.invoke({"question": "Magkano ang LTO renewal?", "rewrite_count": 0})
    # Rewording can't surface what isn't in the corpus, so don't spend a retry.
    assert final_state.get("rewrite_count", 0) == 0
    assert "answer" in final_state


def test_crag_graph_falls_back_to_heuristic_on_bad_grader_response():
    retriever = FakeRetriever(score=0.9)
    app = build_crag_graph(retriever, StubGenerator(), LLMGrader(StubGenerator("not json")))
    final_state = app.invoke({"question": "What are the penalties?", "rewrite_count": 0})
    assert final_state["grade"] == "strong"  # heuristic verdict on a 0.9 score
    assert final_state["verdict"] == ""


def test_format_context_joins_multiple_chunks_with_bracketed_titles():
    retriever = FakeRetriever(
        score=0.9,
        extra_chunks=[
            {"chunk_id": "docB__0", "doc_id": "docB", "title": "Doc B", "text": "Doc B content about deadlines."}
        ],
    )
    retrieved = [
        RetrievedChunk(chunk_id="docA__0", score=0.9, rank=0),
        RetrievedChunk(chunk_id="docB__0", score=0.8, rank=1),
    ]
    context = _format_context(retriever, retrieved)
    assert context == (
        "[Doc A] Doc A content about penalties.\n\n"
        "[Doc B] Doc B content about deadlines."
    )


def test_verify_node_matches_any_of_multiple_retrieved_titles():
    # Citing only the second of two retrieved documents should still verify --
    # verify_node's check is "any", not "all" or "the first one".
    retriever = FakeRetriever(
        score=0.9,
        title="RMC No. 30-2026 Digest",
        extra_chunks=[
            {"chunk_id": "docB__0", "doc_id": "docB", "title": "RMC No. 36-2026", "text": "Extended deadline text."}
        ],
    )
    app = build_crag_graph(retriever, StubGenerator("The new deadline is per [RMC No. 36-2026]."))
    final_state = app.invoke({"question": "What's the deadline?", "rewrite_count": 0})
    assert final_state["citations_verified"] is True


def test_verify_node_true_positive_full_title_cited():
    retriever = FakeRetriever(score=0.9, title="RMC No. 30-2026 Digest")
    answer = "The deadline is May 15, 2026.\n\nSource: **[RMC No. 30-2026 Digest]**"
    app = build_crag_graph(retriever, StubGenerator(answer))
    final_state = app.invoke({"question": "What's the deadline?", "rewrite_count": 0})
    assert final_state["citations_verified"] is True


def test_verify_node_rejects_first_word_only_match():
    # Regression test: title.split()[0] alone used to satisfy the old check,
    # so an answer that only mentions "RMC" generically (never citing this
    # specific document) must NOT count as a verified citation.
    retriever = FakeRetriever(score=0.9, title="RMC No. 30-2026 Digest")
    answer = "Under an RMC issued by the BIR, penalties may apply for late filing."
    app = build_crag_graph(retriever, StubGenerator(answer))
    final_state = app.invoke({"question": "What's the deadline?", "rewrite_count": 0})
    assert final_state["citations_verified"] is False


def test_verify_node_is_case_and_whitespace_insensitive():
    retriever = FakeRetriever(score=0.9, title="RMC No. 30-2026 Digest")
    answer = "See   rmc no. 30-2026   digest for details."
    app = build_crag_graph(retriever, StubGenerator(answer))
    final_state = app.invoke({"question": "What's the deadline?", "rewrite_count": 0})
    assert final_state["citations_verified"] is True


def test_naive_rag_never_rewrites():
    retriever = FakeRetriever(score=0.01)
    state = run_naive_rag(retriever, "Ano ang parusa?", StubGenerator())
    assert "rewrite_count" not in state or state.get("rewrite_count", 0) == 0
    assert "answer" in state
