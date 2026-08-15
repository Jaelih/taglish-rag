from taglish_rag.generation.generator import MockGenerator
from taglish_rag.generation.judge import LLMJudge, _extract_json


def test_mock_generator_refuses_with_no_context():
    gen = MockGenerator()
    result = gen.generate("What is X?", context="")
    assert "don't know" in result.answer.lower()
    assert result.backend == "mock"


def test_mock_generator_echoes_context_when_present():
    gen = MockGenerator()
    result = gen.generate("What is X?", context="X is a government program.")
    assert "X is a government program" in result.answer


def test_extract_json_handles_surrounding_prose():
    text = 'Sure, here is my answer:\n{"groundedness": 0.8, "is_refusal": false}\nHope that helps!'
    parsed = _extract_json(text)
    assert parsed == {"groundedness": 0.8, "is_refusal": False}


def test_extract_json_returns_none_for_garbage():
    assert _extract_json("not json at all") is None


def test_llm_judge_marks_unparseable_response():
    class BrokenGenerator(MockGenerator):
        def generate(self, question, context):
            from taglish_rag.generation.generator import GenerationResult

            return GenerationResult(answer="not json", backend="mock", model="mock")

    judge = LLMJudge(BrokenGenerator())
    score = judge.score("q", "ctx", "ref", "ans")
    assert score.parse_ok is False
    assert score.groundedness is None


def test_llm_judge_parses_well_formed_response():
    class JsonGenerator(MockGenerator):
        def generate(self, question, context):
            from taglish_rag.generation.generator import GenerationResult

            return GenerationResult(
                answer='{"groundedness": 0.9, "correctness": 1.0, "citation_accuracy": 0.5, '
                '"is_refusal": false, "rationale": "looks right"}',
                backend="mock",
                model="mock",
            )

    judge = LLMJudge(JsonGenerator())
    score = judge.score("q", "ctx", "ref", "ans")
    assert score.parse_ok is True
    assert score.groundedness == 0.9
    assert score.is_refusal is False
