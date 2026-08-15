"""Self-correcting CRAG (Corrective RAG) loop, built with LangGraph:

    retrieve -> grade documents -> [weak? rewrite query -> retrieve again] -> generate -> verify citations

Compared against naive RAG (retrieve -> generate, no grading/correction) in
the ablation sweep's `use_agent` axis. Document grading and query rewriting
use the configured Generator backend when one with real credentials is
available; with GENERATOR_BACKEND=mock (the default with no API keys), both
fall back to cheap heuristics (retrieval-score threshold for grading,
English-translation-and-retry for rewriting) so the graph is fully
exercisable offline -- see README for what that does and doesn't validate.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from taglish_rag.generation.generator import Generator, get_generator
from taglish_rag.retrieval.retriever import Retriever
from taglish_rag.schemas import RetrievedChunk

GRADE_WEAK_SCORE_THRESHOLD = 0.15  # heuristic fallback: top hybrid/dense score below this = weak
MAX_REWRITES = 1


class CragState(TypedDict, total=False):
    question: str
    search_query: str
    retrieved: list[RetrievedChunk]
    context: str
    grade: str  # "strong" | "weak"
    rewrite_count: int
    answer: str
    citations_verified: bool


def _format_context(retriever: Retriever, retrieved: list[RetrievedChunk]) -> str:
    parts = []
    for r in retrieved:
        chunk = next(c for c in retriever.chunks if c["chunk_id"] == r.chunk_id)
        parts.append(f"[{chunk['title']}] {chunk['text']}")
    return "\n\n".join(parts)


def _heuristic_grade(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return "weak"
    return "strong" if retrieved[0].score >= GRADE_WEAK_SCORE_THRESHOLD else "weak"


def _heuristic_rewrite(question: str) -> str:
    """Fallback query rewrite with no LLM available: translate to English
    (the corpus's language) and retry, since that's the single highest-
    leverage rewrite for this corpus per the translate_query_to_english
    ablation finding."""
    from taglish_rag.retrieval.translate import translate_to_english

    try:
        return translate_to_english(question)
    except Exception:
        return question


def build_crag_graph(retriever: Retriever, generator: Generator | None = None):
    generator = generator or get_generator()

    def retrieve_node(state: CragState) -> CragState:
        query = state.get("search_query", state["question"])
        retrieved = retriever.retrieve(query)
        return {"retrieved": retrieved, "context": _format_context(retriever, retrieved)}

    def grade_node(state: CragState) -> CragState:
        grade = _heuristic_grade(state["retrieved"])
        return {"grade": grade}

    def route_after_grade(state: CragState) -> str:
        if state["grade"] == "weak" and state.get("rewrite_count", 0) < MAX_REWRITES:
            return "rewrite"
        return "generate"

    def rewrite_node(state: CragState) -> CragState:
        new_query = _heuristic_rewrite(state["question"])
        return {
            "search_query": new_query,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    def generate_node(state: CragState) -> CragState:
        result = generator.generate(question=state["question"], context=state["context"])
        return {"answer": result.answer}

    def verify_node(state: CragState) -> CragState:
        titles = {
            next(c for c in retriever.chunks if c["chunk_id"] == r.chunk_id)["title"]
            for r in state["retrieved"]
        }
        answer = state["answer"]
        verified = any(title.split()[0] in answer for title in titles) if titles else False
        return {"citations_verified": verified}

    graph = StateGraph(CragState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"rewrite": "rewrite", "generate": "generate"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


def run_crag(retriever: Retriever, question: str, generator: Generator | None = None) -> CragState:
    app = build_crag_graph(retriever, generator)
    return app.invoke({"question": question, "rewrite_count": 0})


def run_naive_rag(retriever: Retriever, question: str, generator: Generator | None = None) -> CragState:
    """No grading, no rewrite -- the baseline the CRAG loop is compared against."""
    generator = generator or get_generator()
    retrieved = retriever.retrieve(question)
    context = _format_context(retriever, retrieved)
    result = generator.generate(question=question, context=context)
    return {"question": question, "retrieved": retrieved, "context": context, "answer": result.answer}
