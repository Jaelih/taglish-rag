"""Retrieval evaluator for the CRAG loop: given a question and the passages
retrieval returned, decide whether they are actually sufficient to answer it.

This is the component `agent/crag.py:_heuristic_grade` stands in for. That
fallback thresholds a single raw retrieval score at 0.15, which is fast and
keyless but measures *similarity*, not *sufficiency* -- and its scale shifts
with retrieval mode (fused hybrid vs. bare cosine vs. cross-encoder logit),
so the same threshold means different things across the ablation axes.

Keep `_heuristic_grade` as the offline path (unit tests and any run without
GOOGLE_API_KEY); use this when a Generator is available. Like judge.py, this
passes the whole rendered prompt as the Generator's `question` with an empty
`context`, and degrades rather than crashes on an unparseable response.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from taglish_rag.generation.generator import Generator

Verdict = Literal["correct", "ambiguous", "incorrect"]

GRADER_PROMPT = """You are grading a retrieval system for a question-answering service about \
Philippine government services (BIR taxes, PhilHealth, Pag-IBIG Fund).

You are NOT answering the question. You are judging whether the RETRIEVED PASSAGES contain \
enough information to answer it.

QUESTION: {question}

RETRIEVED PASSAGES:
{passages}

Apply all of these rules:

1. LANGUAGE MISMATCH IS NEVER A REASON TO FAIL A PASSAGE. The question may be in English, \
Tagalog, or Taglish (code-switched English/Tagalog). The document corpus is entirely in \
English. An English passage that answers a Tagalog question is fully sufficient. Judge \
meaning, never surface word overlap.

2. TOPICAL SIMILARITY IS NOT SUFFICIENCY. A passage on the right subject that does not state \
the specific fact asked for is not sufficient. Text describing ITR filing penalties in general \
does not answer "how much is the penalty for late filing".

3. CHECK THE PERIOD. If the question asks for a current or year-specific value (a rate, a \
deadline, a dividend, a contribution table) and the passages only give that value for a \
different, expired, or superseded period, they are NOT sufficient -- even though they look \
like a direct match. Say so in "missing".

4. CHECK THE AGENCY. If the question is about an agency not covered by the passages (for \
example SSS or LTO, when the passages are BIR, PhilHealth, or Pag-IBIG), the passages are not \
sufficient no matter how topically close they read.

5. MULTI-PART QUESTIONS need every part covered. If the passages answer one half of a \
two-part question, that is "ambiguous", not "correct".

6. JUDGE ONLY WHAT IS WRITTEN IN THE PASSAGES. Do not use your own knowledge of Philippine \
policy to fill a gap, and do not assume an unshown part of a document contains the answer.

Respond with ONLY a JSON object, no other text:
{{
  "verdict": "<correct | ambiguous | incorrect>",
  "sufficient_passages": [<1-based indices of the passages that carry the answer; [] if none>],
  "missing": "<one sentence naming what is absent, or \\"\\" if verdict is correct>",
  "corpus_gap": <true if the passages indicate this corpus simply does not cover the question \
-- wrong agency, or the value exists only for a period outside what the question asks; false \
if the answer plausibly exists in the corpus and retrieval just failed to surface it>
}}

Verdict meanings:
- "correct": at least one passage directly supports a complete answer.
- "ambiguous": partial support -- related and useful, but incomplete, or you cannot tell \
whether it is current or in-scope.
- "incorrect": nothing here supports an answer.

"corpus_gap" is a routing signal, not a quality score: it says whether retrying retrieval \
with a reworded query could plausibly help. Set it true when rewording cannot help because \
the information is not in this corpus at all."""


class GradeResult(BaseModel):
    """One grading decision. `parse_ok=False` means the model's response
    wasn't usable JSON; callers should fall back to `_heuristic_grade`
    rather than trust the defaults below."""

    verdict: Verdict = "ambiguous"
    sufficient_passages: list[int] = Field(default_factory=list)
    missing: str = ""
    corpus_gap: bool = False
    parse_ok: bool = True

    @property
    def grade(self) -> str:
        """Collapse the three-way verdict onto the binary strong/weak the
        CRAG graph currently routes on. Only "correct" is strong -- an
        "ambiguous" verdict means we couldn't confirm sufficiency, which is
        exactly the case a rewrite-and-retry exists for."""
        return "strong" if self.verdict == "correct" else "weak"


def format_passages(texts: list[str]) -> str:
    """Number passages 1-based so the model's `sufficient_passages` indices
    line up with the retrieved list's order."""
    if not texts:
        return "(no passages retrieved)"
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(texts, start=1))


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class LLMGrader:
    """Retrieval evaluator backed by a Generator. Takes the generator by
    injection -- same as LLMJudge -- so it reuses the caller's client and
    therefore its rate limiter and retry policy instead of opening a second
    one (see generation/generator.py)."""

    def __init__(self, generator: Generator):
        self._generator = generator

    def grade(self, question: str, passages: list[str]) -> GradeResult:
        if not passages:
            # Nothing retrieved: no need to spend a call to learn that.
            return GradeResult(verdict="incorrect", missing="no passages were retrieved")

        prompt = GRADER_PROMPT.format(question=question, passages=format_passages(passages))
        result = self._generator.generate(question=prompt, context="")
        parsed = _extract_json(result.answer)
        if parsed is None:
            return GradeResult(missing="unparseable grader response", parse_ok=False)

        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in ("correct", "ambiguous", "incorrect"):
            return GradeResult(missing=f"unrecognized verdict {verdict!r}", parse_ok=False)

        indices = parsed.get("sufficient_passages") or []
        if not isinstance(indices, list):
            indices = []

        return GradeResult(
            verdict=verdict,
            # Drop anything out of range: the indices are used to slice the
            # retrieved list, and the model occasionally invents one.
            sufficient_passages=[i for i in indices if isinstance(i, int) and 1 <= i <= len(passages)],
            missing=str(parsed.get("missing", "")),
            corpus_gap=bool(parsed.get("corpus_gap", False)),
        )
