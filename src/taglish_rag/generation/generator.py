"""Answer generation backed by Google AI Studio (Gemini Flash), which has
a genuinely free tier with no card required. `GOOGLE_API_KEY` must be set
in .env -- retrieval, ingestion, and the retrieval eval all still run
fully offline without it, but answer generation, the CRAG loop's generate
step, and the LLM-as-judge all need a live key.
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
    #: Short provider label stamped into GenerationResult and into eval
    #: result files, so a run always records which backend produced it.
    backend: str = "unknown"

    def generate(self, question: str, context: str) -> GenerationResult:
        raise NotImplementedError


class GeminiGenerator(Generator):
    backend = "gemini"

    def __init__(self, model: str = "gemini-3.1-flash-lite"):
        from google import genai

        api_key = env("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model

    def generate(self, question: str, context: str) -> GenerationResult:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self._model_name,
            contents=f"Context:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        return GenerationResult(answer=resp.text, backend=self.backend, model=self._model_name)


def get_generator() -> Generator:
    return GeminiGenerator()
