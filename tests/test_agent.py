from conftest import StubGenerator

from taglish_rag.agent.crag import _heuristic_grade, build_crag_graph, run_naive_rag
from taglish_rag.schemas import RetrievedChunk


class FakeRetriever:
    """Minimal stand-in for taglish_rag.retrieval.retriever.Retriever, so
    the CRAG graph's routing logic can be tested without loading real
    chunk files or embedding models."""

    def __init__(self, score: float):
        self._score = score
        self.chunks = [
            {"chunk_id": "docA__0", "doc_id": "docA", "title": "Doc A", "text": "Doc A content about penalties."}
        ]

    def retrieve(self, query: str):
        return [RetrievedChunk(chunk_id="docA__0", score=self._score, rank=0)]


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


def test_naive_rag_never_rewrites():
    retriever = FakeRetriever(score=0.01)
    state = run_naive_rag(retriever, "Ano ang parusa?", StubGenerator())
    assert "rewrite_count" not in state or state.get("rewrite_count", 0) == 0
    assert "answer" in state
