# What I learned measuring Taglish retrieval

*Draft — fill in bracketed numbers from `results/ablation_summary.json` before publishing.*

Every RAG portfolio project says "I built a retrieval system." Almost none of them say how they know it works. I wanted to fix that for myself, so I built Taglish-RAG: a QA system over Philippine government documents (BIR taxes, PhilHealth, Pag-IBIG Fund), and spent as much effort on *measuring* it as building it.

## The setup

83 real government documents — tax circulars, health-insurance FAQs, housing-loan terms — all in English, because that's the language Philippine government agencies publish in. But that's not the language most Filipinos ask questions in. I wrote 90 questions three ways: English, Tagalog, and Taglish (the code-switched mix most people actually type), asking the exact same thing each time. Then I measured whether retrieval quality held up as the query language drifted away from the corpus's language.

## The headline number

[FILL IN: e.g. "Plain BM25 keyword search hit the right document on the first try 72% of the time for English questions — and just 3% of the time for the identical question in Tagalog."] That's not a subtle degradation, it's the retrieval system nearly failing outright for one of the three languages it needs to serve, using the single most common retrieval method in RAG tutorials.

[FILL IN: how much of that gap dense embeddings and hybrid retrieval closed, from results/ablation_summary.json's retrieval_mode and embedding_model axes]

## What actually helped (and what didn't)

Ran a 6-axis ablation — embedding model, chunk size, dense vs. BM25 vs. hybrid, reranking on/off, query translation on/off, naive RAG vs. a self-correcting CRAG agent — holding everything else fixed at a baseline config. [FILL IN 2-3 sentence honest summary once results are in, including anything that *didn't* help — a reranker or query-translation step that turned out not to matter is a more interesting finding than a clean win, because it says something about where the bottleneck actually is.]

## The part I think matters most: the unanswerable questions

Eighteen of the 90 questions are deliberately unanswerable from the corpus — and several of those are adversarial on purpose. One asks for the *current* BIR percentage tax rate; the only relevant document in the corpus describes a temporary rate that expired in 2023. Another asks for the 2026 Pag-IBIG dividend rate; the corpus's rate table stops at 2025. A system that retrieves the nearest-sounding chunk and answers confidently — without checking whether that chunk is actually still current — will get these wrong in a way that looks completely plausible. That's the failure mode that matters in production, and it's the one most portfolio RAG projects never test for.

## Honest limitations

- The corpus is a snapshot, not a live feed — several documents describe time-bound policy that has since changed (see above; used deliberately for the adversarial questions, but worth stating).
- 90 questions, one annotator. Judge-vs-human agreement (Cohen's κ) on a 30-item sample is in `eval/human_labels/` once a generation backend is configured — I'd rather ship that measurement honestly incomplete than fabricate it.
- Ran the full ablation sweep on a local CPU, not the free Colab GPU the original plan called for — bge-m3 and multilingual-e5-large are large enough that this made the sweep slow rather than infeasible, but a GPU would make iterating on this much faster next time.

Repo: [link] · Demo: [link] · Dataset: [link]
