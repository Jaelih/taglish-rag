import json

from taglish_rag.config import REPO_ROOT
from taglish_rag.schemas import EvalItem

EVAL_PATH = REPO_ROOT / "eval" / "taglish_rag_eval_v1.jsonl"


def _load_items():
    with open(EVAL_PATH, encoding="utf-8") as f:
        return [EvalItem(**json.loads(line)) for line in f if line.strip()]


def _load_known_doc_ids():
    doc_ids = set()
    for agency in ["bir", "philhealth", "pagibig"]:
        manifest = REPO_ROOT / "data" / "raw" / agency / "manifest.jsonl"
        if not manifest.exists():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                doc_ids.add(json.loads(line)["doc_id"])
    return doc_ids


def test_eval_set_qids_are_unique():
    items = _load_items()
    qids = [i.qid for i in items]
    assert len(qids) == len(set(qids))


def test_eval_set_gold_doc_ids_exist_in_corpus():
    items = _load_items()
    known = _load_known_doc_ids()
    if not known:
        return  # scrapers haven't been run in this environment; skip
    for item in items:
        for doc_id in item.gold_doc_ids:
            assert doc_id in known, f"{item.qid} cites unknown doc_id {doc_id}"


def test_eval_set_unanswerable_items_have_no_gold_docs():
    for item in _load_items():
        if item.question_type == "unanswerable":
            assert item.gold_doc_ids == []
