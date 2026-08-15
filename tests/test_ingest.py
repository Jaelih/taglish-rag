from taglish_rag.ingest.chunk import chunk_text
from taglish_rag.ingest.clean import clean_corpus, find_boilerplate_lines, normalize_whitespace


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # consecutive chunks should share some words (the overlap region)
    shared = set(chunks[0].split()) & set(chunks[1].split())
    assert len(shared) > 0


def test_chunk_text_rejects_overlap_ge_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=10, overlap=10)


def test_chunk_text_empty_string():
    assert chunk_text("", chunk_size=100, overlap=10) == []


def test_normalize_whitespace_collapses_blank_lines():
    assert normalize_whitespace("a\n\n\n\nb") == "a\n\nb"


def test_find_boilerplate_lines_detects_repeated_nav():
    docs = [
        "Unique intro A\nHome | About | Contact\nBody text A",
        "Unique intro B\nHome | About | Contact\nBody text B",
        "Unique intro C\nHome | About | Contact\nBody text C",
    ]
    boilerplate = find_boilerplate_lines(docs)
    assert "Home | About | Contact" in boilerplate
    assert "Unique intro A" not in boilerplate


def test_clean_corpus_strips_boilerplate_but_keeps_unique_content():
    docs = {
        "d1": "Intro one\nHome | About | Contact\nBody one",
        "d2": "Intro two\nHome | About | Contact\nBody two",
        "d3": "Intro three\nHome | About | Contact\nBody three",
    }
    cleaned = clean_corpus(docs)
    for doc_id, text in cleaned.items():
        assert "Home | About | Contact" not in text
        assert "Body" in text
