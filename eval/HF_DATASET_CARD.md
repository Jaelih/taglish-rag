---
language:
- en
- tl
license: cc-by-4.0
task_categories:
- question-answering
pretty_name: Taglish-RAG Eval Set v1
size_categories:
- n<1K
---

# Taglish-RAG: A Retrieval & Evaluation Benchmark for Code-Switched Filipino QA

90 question–answer pairs for evaluating RAG systems on **code-switched Filipino (Taglish) questions** about Philippine public services — BIR (tax), PhilHealth (health insurance), and Pag-IBIG Fund (housing/savings) — answered from a corpus of 83 official government documents.

Each question exists in **3 parallel versions** (English / Tagalog / Taglish) asking the same underlying thing, against a corpus that is entirely **English-language** source text — this is the retrieval challenge the benchmark targets: how much does retrieval quality degrade for the same information need as the query code-switches away from the corpus's language?

## Dataset structure

```json
{
  "qid": "bir-01-en",
  "question": "What is the new deadline for filing the 2025 Annual Income Tax Return...",
  "language": "en",
  "question_type": "factual",
  "gold_answer": "May 15, 2026 (extended from the original April 15, 2026 deadline...)",
  "gold_doc_ids": ["bir-rmc-no-30-2026", "bir-rmc-no-36-2026"],
  "agency": "bir"
}
```

- **`question_type`**: `factual` (single-topic lookup, n=39) / `multi_hop` (requires combining ≥2 facts, n=21) / `unanswerable` (deliberately not covered by the corpus — tests refusal/hallucination, n=18) / `ambiguous` (underspecified — a correct system should ask for clarification, n=12).
- **`gold_doc_ids`** cites source *documents*, not chunks, so the eval set stays valid regardless of how a downstream system chunks the corpus.
- Full methodology, including how the adversarial `unanswerable` items were designed (some cite a real, retrievable, topically-adjacent-but-wrong document — see `DATASHEET.md`), is in the companion [DATASHEET.md](./DATASHEET.md).

## Companion corpus

The source documents (`gold_doc_ids` point here) are in the parent repository's `data/raw/{bir,philhealth,pagibig}/` — see [the GitHub repo] for the full corpus, ingestion pipeline, and retrieval/generation evaluation harness this eval set is designed to drive.

## License

Question text and gold answers: original authorship, CC-BY-4.0. Source documents are Philippine government works, public domain per IP Code §176.

## Citation

If you use this benchmark, please cite the repository (see the main README for the current citation format).
