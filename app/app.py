"""Gradio demo: ask a question in English, Tagalog, or Taglish about BIR,
PhilHealth, or Pag-IBIG; see the answer plus which source documents it
cites. Deployed to HuggingFace Spaces (see README `## Demo`).
"""
from __future__ import annotations

import gradio as gr

from taglish_rag.agent.crag import run_crag, run_naive_rag
from taglish_rag.generation.generator import get_generator
from taglish_rag.retrieval.retriever import Retriever, RetrievalConfig
from taglish_rag.agent.grader import LLMGrader

_retriever = None
_generator = None
_grader = None

def _lazy_init():
    global _retriever, _generator, _grader
    if _retriever is None:
        _retriever = Retriever(RetrievalConfig(embedding_model="minilm-multilingual"))
        _generator = get_generator()
        _grader = LLMGrader(_generator)

    return _retriever, _generator, _grader


def ask(question: str, use_agent: bool) -> tuple[str, str]:
    if not question.strip():
        return "", ""
    retriever, generator, grader = _lazy_init()
    if use_agent:
        state = run_crag(retriever, question, generator, grader)
    else:
        state = run_naive_rag(retriever, question, generator)

    answer = state.get("answer", "(no answer)")
    seen_titles = []
    for r in state.get("retrieved", []):
        chunk = retriever.chunks_by_id[r.chunk_id]
        entry = f"- **{chunk['title']}** ({chunk['agency'].upper()}) — {chunk['url']}"
        if entry not in seen_titles:
            seen_titles.append(entry)
    citations = "\n".join(seen_titles) if seen_titles else "_(no sources retrieved)_"
    return answer, citations


EXAMPLES = [
    ["Magkano ang penalty kung late mag-file ng ITR?", False],
    ["What documents do I need to claim Pag-IBIG MP2 savings?", False],
    ["Sino ang qualified maging PhilHealth Senior Citizen member?", True],
    ["Ano ang current SSS contribution rate?", False],
]

with gr.Blocks(title="Taglish-RAG") as demo:
    gr.Markdown(
        "# Taglish-RAG\n"
        "Ask about **BIR**, **PhilHealth**, or **Pag-IBIG** in English, Tagalog, or Taglish. "
        "Answers are generated only from the retrieved government documents shown below — "
        "if nothing relevant is retrieved, the system is expected to say so rather than guess."
    )
    with gr.Row():
        question = gr.Textbox(label="Your question", placeholder="Magkano ang penalty kung late mag-file ng ITR?", scale=4)
        use_agent = gr.Checkbox(label="Use CRAG agent (self-correcting)", value=False, scale=1)
    ask_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Markdown(label="Answer")
    citations_box = gr.Markdown(label="Sources retrieved")

    ask_btn.click(ask, inputs=[question, use_agent], outputs=[answer_box, citations_box])
    question.submit(ask, inputs=[question, use_agent], outputs=[answer_box, citations_box])
    # gr.Examples(examples=EXAMPLES, inputs=[question, use_agent])

if __name__ == "__main__":
    demo.launch()
