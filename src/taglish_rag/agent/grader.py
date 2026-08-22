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
