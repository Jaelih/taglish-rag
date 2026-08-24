"""Self-correcting CRAG (Corrective RAG) loop, built with LangGraph:

    retrieve -> grade documents -> [weak? rewrite query -> retrieve again] -> generate -> verify citations

Compared against naive RAG (retrieve -> generate, no grading/correction) in
the ablation sweep's `use_agent` axis. Document grading and query rewriting
currently use cheap heuristics rather than the Generator -- a retrieval-score
threshold for grading, English-translation-and-retry for rewriting -- which
keeps the graph's routing logic fast, deterministic, and unit-testable
without an API key. Only the generate step calls the LLM. See README for
what that does and doesn't validate.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from taglish_rag.agent.grader import LLMGrader
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
    verdict: str  # "correct" | "ambiguous" | "incorrect"; "" when heuristically graded
    corpus_gap: bool
    grade_missing: str
    rewrite_count: int
    answer: str
    citations_verified: bool


def _normalize_for_match(text: str) -> str:
    """Case/whitespace normalization so a title-in-answer check isn't tripped
    up by markdown bolding, extra spaces, or case drift."""
    return " ".join(text.lower().split())


def _format_context(retriever: Retriever, retrieved: list[RetrievedChunk]) -> str:
    parts = []
    for r in retrieved:
        chunk = retriever.chunks_by_id[r.chunk_id]
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


def build_crag_graph(
    retriever: Retriever,
    generator: Generator | None = None,
    grader: LLMGrader | None = None,
):
    """`grader` is the CRAG paper's retrieval evaluator. Pass one to grade on
    sufficiency; leave it None to keep the keyless score-threshold heuristic
    (what the unit tests and any run without GOOGLE_API_KEY use)."""
    generator = generator or get_generator()

    def retrieve_node(state: CragState) -> CragState:
        query = state.get("search_query", state["question"])
        retrieved = retriever.retrieve(query)
        return {"retrieved": retrieved, "context": _format_context(retriever, retrieved)}

    def grade_node(state: CragState) -> CragState:
        retrieved = state["retrieved"]
        if grader is not None:
            passages = [retriever.texts_by_id[r.chunk_id] for r in retrieved]
            result = grader.grade(state["question"], passages)
            if result.parse_ok:
                return {
                    "grade": result.grade,
                    "verdict": result.verdict,
                    "corpus_gap": result.corpus_gap,
                    "grade_missing": result.missing,
                }
            # Unusable grader response -- fall through to the heuristic
            # rather than let one bad completion decide the route.
        return {"grade": _heuristic_grade(retrieved), "verdict": "", "corpus_gap": False}

    def route_after_grade(state: CragState) -> str:
        # A corpus gap means the answer isn't in this corpus at all, so
        # re-retrieving a reworded query against the same corpus can't help.
        if state.get("corpus_gap"):
            return "generate"
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
        titles = {retriever.chunks_by_id[r.chunk_id]["title"] for r in state["retrieved"]}
        # SYSTEM_PROMPT asks the model to "cite which source titles you used,"
        # and it does so by reproducing the title verbatim (e.g. "[RMC No.
        # 30-2026 Digest]"), so require the whole normalized title rather than
        # just its first word -- titles starting "RMC"/"PhilHealth"/"Circular"
        # would otherwise match almost any answer that mentions the agency.
        answer_norm = _normalize_for_match(state["answer"])
        verified = any(_normalize_for_match(title) in answer_norm for title in titles) if titles else False
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


def run_crag(
    retriever: Retriever,
    question: str,
    generator: Generator | None = None,
    grader: LLMGrader | None = None,
) -> CragState:
    app = build_crag_graph(retriever, generator, grader)
    return app.invoke({"question": question, "rewrite_count": 0})


def run_naive_rag(retriever: Retriever, question: str, generator: Generator | None = None) -> CragState:
    """No grading, no rewrite -- the baseline the CRAG loop is compared against."""
    generator = generator or get_generator()
    retrieved = retriever.retrieve(question)
    context = _format_context(retriever, retrieved)
    result = generator.generate(question=question, context=context)
    return {"question": question, "retrieved": retrieved, "context": context, "answer": result.answer}
