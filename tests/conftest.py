from taglish_rag.generation.generator import GenerationResult, Generator


class StubGenerator(Generator):
    """Offline test double for the Gemini-backed generator, so the CRAG
    graph and the judge stay unit-testable with no API key configured
    (CI runs keyless)."""

    backend = "stub"

    def __init__(self, answer: str = "stub answer"):
        self._answer = answer

    def generate(self, question: str, context: str) -> GenerationResult:
        return GenerationResult(answer=self._answer, backend=self.backend, model="stub")
