"""LLM-as-judge for generation quality: groundedness, answer correctness,
citation accuracy, and refusal accuracy on deliberately unanswerable
questions. Uses the same pluggable Generator backend as answer generation
(judge model configurable independently in practice by pointing
GENERATOR_BACKEND at a different provider for the judge call).

Judge output is parsed as JSON; a malformed/unparseable response degrades
to a `None` score rather than crashing the eval run, and is counted
separately so judge reliability itself is visible in the results.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from taglish_rag.generation.generator import Generator

JUDGE_PROMPT_TEMPLATE = """You are evaluating a RAG system's answer to a question about Philippine \
government services (BIR, PhilHealth, or Pag-IBIG). Score the SYSTEM ANSWER against the \
RETRIEVED CONTEXT and the REFERENCE ANSWER.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

REFERENCE ANSWER (may be "not answerable" if this is deliberately an unanswerable question):
{reference}

SYSTEM ANSWER:
{answer}

Score on these axes and respond with ONLY a JSON object, no other text:
{{
  "groundedness": <0.0-1.0, does every claim in the system answer trace back to the retrieved context?>,
  "correctness": <0.0-1.0, does the system answer match the reference answer's meaning?>,
  "citation_accuracy": <0.0-1.0, are any cited sources actually the right ones?>,
  "is_refusal": <true/false, did the system answer decline to answer / say it doesn't know?>,
  "rationale": "<one sentence>"
}}"""


@dataclass
class JudgeScore:
    groundedness: float | None
    correctness: float | None
    citation_accuracy: float | None
    is_refusal: bool | None
    rationale: str
    parse_ok: bool


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class LLMJudge:
    def __init__(self, generator: Generator):
        self._generator = generator

    def score(self, question: str, context: str, reference: str, answer: str) -> JudgeScore:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question, context=context or "(none)", reference=reference, answer=answer
        )
        result = self._generator.generate(question=prompt, context="")
        parsed = _extract_json(result.answer)
        if parsed is None:
            return JudgeScore(None, None, None, None, "unparseable judge response", parse_ok=False)
        return JudgeScore(
            groundedness=parsed.get("groundedness"),
            correctness=parsed.get("correctness"),
            citation_accuracy=parsed.get("citation_accuracy"),
            is_refusal=parsed.get("is_refusal"),
            rationale=parsed.get("rationale", ""),
            parse_ok=True,
        )
