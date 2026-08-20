from conftest import StubGenerator

from taglish_rag.generation.judge import LLMJudge, _extract_json


def test_extract_json_handles_surrounding_prose():
    text = 'Sure, here is my answer:\n{"groundedness": 0.8, "is_refusal": false}\nHope that helps!'
    parsed = _extract_json(text)
    assert parsed == {"groundedness": 0.8, "is_refusal": False}


def test_extract_json_returns_none_for_garbage():
    assert _extract_json("not json at all") is None


def test_llm_judge_marks_unparseable_response():
    judge = LLMJudge(StubGenerator("not json"))
    score = judge.score("q", "ctx", "ref", "ans")
    assert score.parse_ok is False
    assert score.groundedness is None


def test_llm_judge_parses_well_formed_response():
    judge = LLMJudge(
        StubGenerator(
            '{"groundedness": 0.9, "correctness": 1.0, "citation_accuracy": 0.5, '
            '"is_refusal": false, "rationale": "looks right"}'
        )
    )
    score = judge.score("q", "ctx", "ref", "ans")
    assert score.parse_ok is True
    assert score.groundedness == 0.9
    assert score.is_refusal is False
