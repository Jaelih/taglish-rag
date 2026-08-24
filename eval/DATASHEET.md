# Datasheet: Taglish-RAG Eval Set v1

Following the spirit of [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (Gebru et al.), abbreviated for a benchmark of this size.

## Motivation

**For what purpose was the dataset created?** To evaluate retrieval-augmented QA systems on code-switched Filipino (Taglish) questions about Philippine public services, where source documents are English-language government issuances but real users ask in English, Tagalog, or Taglish. Existing multilingual RAG evals target European language pairs; a Taglish, code-switched, single-language-corpus benchmark was not found publicly available at time of writing.

**Who created it?** Authored by hand (not LLM-generated) grounded in a 83-document corpus scraped from three Philippine government agencies (BIR, PhilHealth, Pag-IBIG Fund), as part of a personal portfolio project.

## Composition

- **90 question–answer pairs**, each with: `qid`, `question`, `language`, `question_type`, `gold_answer`, `gold_doc_ids`, `agency`.
- **Languages**: 30 English / 30 Tagalog / 30 Taglish — each set of 3 is a parallel translation of the same underlying information need, so retrieval/generation quality can be compared *for the same question* across languages.
- **Question types**: 39 factual (single-topic lookup), 21 multi-hop (requires combining ≥2 facts, sometimes from ≥2 documents), 18 unanswerable (deliberately not covered by the corpus), 12 ambiguous (underspecified — a correct system should ask for clarification rather than guess).
- **Agencies**: 30 BIR (tax) / 30 PhilHealth (health insurance) / 30 Pag-IBIG (housing fund), each built from 10 "topics" translated into all 3 languages.

## Collection process

Every factual/multi-hop gold answer is grounded in specific documents from `data/raw/{bir,philhealth,pagibig}/manifest.jsonl`, verified by directly reading the cleaned extracted text (`data/processed/doc_text_clean/`) before writing the question — not generated from memory or an LLM's general knowledge of Philippine government policy. `gold_doc_ids` cites documents, not chunk IDs, so the eval set stays valid across the chunk-size ablation sweep (chunk IDs are config-dependent; document IDs are not).

Unanswerable items are deliberately adversarial: several (`bir-08`, `pg-07`) cite a real, retrievable, topically-adjacent document that contains a superficially similar but *wrong* answer (an expired tax rate, a stale dividend-rate table capped at the prior year) — a system that retrieves the nearest chunk and answers confidently, without checking recency/applicability, will get these wrong. Others (`bir-09`, `ph-09`, `pg-08`) test whether the system recognizes questions about an entirely different government agency (SSS, LTO) not in its corpus, rather than hallucinating from superficial topical similarity (e.g. both Pag-IBIG and SSS are payroll deductions).

## Known limitations

- **90 vs. 120 questions**: the original plan targeted 120 (50 factual / 30 multi-hop / 25 unanswerable / 15 ambiguous). This v1 ships 90, scaled proportionally, prioritizing that every gold answer be independently verified against source text over hitting a round number.
- **Corpus staleness**: the corpus is a snapshot from a single scrape session; several BIR circulars reference rates/deadlines that are themselves time-bound (e.g. a temporary 2020–2023 tax rate). This is intentional — see the unanswerable-question design above — but means gold answers should not be treated as current tax/benefit advice.
- **PhilHealth 2024 circulars**: several 2024 PDF circulars extracted as empty text (scanned/image-based PDFs with no text layer); no eval questions cite them.
- **Single annotator**: all 90 items were authored by one person in one pass. No inter-annotator agreement was computed for the gold answers themselves (only for the LLM-judge vs. human agreement — see `eval/human_labels/`).

## Distribution

Released under the same terms as the corpus: Philippine government works are public domain per IP Code Sec. 176 (§176). The question text and gold answers are original authorship, released under CC-BY-4.0 (see `eval/LICENSE`).
