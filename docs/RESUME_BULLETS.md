# Resume bullets (fill in the bracketed numbers once results/ablation_summary.json is final)

Replace the weakest existing bullet (per the plan, that's the Image Classification / TinyVGG line — it reads as coursework) with:

> **Taglish-RAG — Multilingual RAG Benchmark & System** · [demo] · [repo] · [dataset]
> Built and open-sourced a 90-question code-switched Filipino QA benchmark and an evaluation harness measuring retrieval (Recall@k, MRR, nDCG@10) and generation (groundedness, citation accuracy, hallucination rate on unanswerable queries). Scraped and cleaned an 83-document corpus from 3 Philippine government agencies; ran a 6-axis ablation across embedding models, chunk size, hybrid BM25+dense retrieval, reranking, and query translation — [hybrid retrieval improved Taglish Recall@5 by X% over BM25-only / dense embeddings closed Y% of the English-vs-Tagalog retrieval gap that pure keyword search missed entirely]. Deployed via a LangGraph self-correcting CRAG loop, Docker, and CI-gated evals on HuggingFace Spaces.

Fill in the bracketed clause from `results/ablation_summary.json` — the honest headline finding, whichever direction it points. Two real candidates already visible from smoke testing during the build (verify against the final full sweep before using either in print):
- BM25-only Taglish/Tagalog Recall@1 was near-zero (~3%) against the English-only corpus, while English Recall@1 was ~72% — the raw magnitude of the code-switching retrieval gap.
- Hybrid (BM25 + dense) retrieval outperformed either alone on MRR/nDCG@10 overall, while dense-only embeddings recovered much more of the Tagalog/Taglish gap than BM25 alone did.

**Also fix the Amdocs bullet** — add whatever numbers can be honestly recalled (document count, users, latency, deflection rate); this project's methodology retroactively legitimizes that bullet by giving you a second, rigorous example of *how* you measure a RAG system, not just that you built one.

**Expand the Skills line**: LangChain, LangGraph, RAG evaluation, vector databases (FAISS, pgvector), hybrid retrieval (BM25 + dense), cross-encoder reranking, LLM-as-judge, HuggingFace, Docker, CI/CD.

## Interview talking points this unlocks

- *"How do you know your RAG system works?"* → a methodology, not a vibe: doc-level gold citations, Recall@k/MRR/nDCG@10, a 6-axis ablation.
- *"What surprised you?"* → the per-language retrieval gap, with the actual EN vs. TL vs. Taglish numbers from `results/ablation_summary.json`.
- *"How do you handle hallucination?"* → measured refusal accuracy on 18 deliberately unanswerable questions, several of which are adversarial (a real, retrievable, topically-adjacent-but-wrong document sits right next to the correct "I don't know").
- *"What are your system's limitations?"* → judge-human κ (once computed, see `eval/human_labels/`), corpus staleness (time-bound tax rates), single-annotator eval set, small corpus (83 docs) — stated proactively in `eval/DATASHEET.md`.
- *"When is an agent worth the latency?"* → CRAG-vs-naive numbers from the generation eval, including honestly reporting it if the answer is "it wasn't."
- *"Tell me about a scraping problem you solved"* → BIR's client-rendered site with no crawlable listings (curated seed list from CDN URLs); Pag-IBIG's bot-gating that blocked `requests`/`curl` but not a real browser session (see `eval/DATASHEET.md` and the README's Corpus section for how each was handled, and what it means for reproducibility).

**Note**: if an ablation shows the fancy option *lost* (e.g. the reranker or query translation didn't help) — report it. "Reranking didn't help and here's my hypothesis why" is a stronger interview signal than a clean win.
