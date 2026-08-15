"""Pluggable answer-generator backends, selected via GENERATOR_BACKEND
(.env). All three genuinely free tiers, no card required: Groq
(Llama 3.3 70B), Google AI Studio (Gemini Flash), or a deterministic
`mock` backend that needs no network/API key at all so the rest of the
pipeline (CRAG loop, judge, eval runner) is exercisable without keys.
"""
from __future__ import annotations

from dataclasses import dataclass

from taglish_rag.config import env

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about Philippine government "
    "services (BIR, PhilHealth, Pag-IBIG) using ONLY the provided context. "
    "If the context does not contain the answer, say clearly that you don't know "
    "rather than guessing. Cite which source titles you used."
)


@dataclass
class GenerationResult:
    answer: str
    backend: str
    model: str


class Generator:
    def generate(self, question: str, context: str) -> GenerationResult:
        raise NotImplementedError


class MockGenerator(Generator):
    """Deterministic, offline stand-in so the pipeline (CRAG loop, judge,
    eval runner) is fully exercisable with zero API keys configured. Not
    a claim of generation quality -- see README for what this does and
    doesn't validate."""

    def generate(self, question: str, context: str) -> GenerationResult:
        if not context.strip():
            answer = "I don't know — no relevant context was retrieved for this question."
        else:
            snippet = context.strip().splitlines()[0][:300]
            answer = (
                f"[mock-generator] Based on the retrieved context, here is the most "
                f"relevant excerpt: \"{snippet}\" -- (a real LLM backend would synthesize "
                f"a full answer from the complete retrieved context)."
            )
        return GenerationResult(answer=answer, backend="mock", model="mock-echo-v1")


class GroqGenerator(Generator):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq

        api_key = env("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self._client = Groq(api_key=api_key)
        self._model = model

    def generate(self, question: str, context: str) -> GenerationResult:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.0,
        )
        return GenerationResult(
            answer=resp.choices[0].message.content, backend="groq", model=self._model
        )


class GeminiGenerator(Generator):
    def __init__(self, model: str = "gemini-1.5-flash"):
        import google.generativeai as genai

        api_key = env("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self._model_name = model
        self._model = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)

    def generate(self, question: str, context: str) -> GenerationResult:
        resp = self._model.generate_content(f"Context:\n{context}\n\nQuestion: {question}")
        return GenerationResult(answer=resp.text, backend="gemini", model=self._model_name)


def get_generator(backend: str | None = None) -> Generator:
    backend = backend or env("GENERATOR_BACKEND", "mock")
    if backend == "groq":
        return GroqGenerator()
    if backend == "gemini":
        return GeminiGenerator()
    if backend == "mock":
        return MockGenerator()
    raise ValueError(f"Unknown GENERATOR_BACKEND '{backend}'. Options: groq, gemini, mock")
