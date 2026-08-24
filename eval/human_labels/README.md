# Human labels for judge-agreement (Cohen's κ)

`sample_qids.txt` is a fixed random sample (seed=42) of 30 qids out of the 90-item eval set, used to measure LLM-judge reliability per the plan ("hand-label 30 items yourself and report judge-human agreement").

**This directory intentionally ships without `human_labels.jsonl`.** Producing it requires:

1. A configured `GOOGLE_API_KEY` (see `.env.example`) and a completed generation eval run (`uv run python -m taglish_rag.eval.generation_runner`) — there's no system answer to label without one.
2. A human (not an LLM) reading each of the 30 sampled system answers against its gold answer and retrieved context, then judging pass/fail.

Fabricating this file would defeat its purpose — it exists to catch cases where the LLM judge is wrong, so it has to reflect an actual independent read.

## To produce it

```
uv run python -m taglish_rag.eval.generation_runner --label baseline
# read results/generation_baseline.json, and for each qid in sample_qids.txt,
# append a line to human_labels.jsonl:
echo '{"qid": "bir-03-tl", "human_acceptable": true}' >> eval/human_labels/human_labels.jsonl
# ...then:
uv run python -m taglish_rag.eval.kappa --generation-result results/generation_baseline.json
```
