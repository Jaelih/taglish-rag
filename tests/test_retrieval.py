from taglish_rag.eval.metrics import mrr, ndcg_at_k, ranked_doc_ids, recall_at_k
from taglish_rag.retrieval.index import fuse_hybrid


def test_ranked_doc_ids_dedups_keeping_first_rank():
    chunk_ids = ["docA__0", "docA__1", "docB__0", "docC__0"]
    doc_id_by_chunk = {
        "docA__0": "docA",
        "docA__1": "docA",
        "docB__0": "docB",
        "docC__0": "docC",
    }
    assert ranked_doc_ids(chunk_ids, doc_id_by_chunk) == ["docA", "docB", "docC"]


def test_recall_at_k_partial_credit_for_multi_doc_gold():
    ranked = ["docA", "docX", "docB", "docY"]
    gold = ["docA", "docB", "docZ"]
    assert recall_at_k(ranked, gold, k=2) == 1 / 3
    assert recall_at_k(ranked, gold, k=4) == 2 / 3


def test_recall_at_k_empty_gold_is_nan():
    result = recall_at_k(["docA"], [], k=5)
    assert result != result  # NaN


def test_mrr_first_hit_rank():
    assert mrr(["docX", "docA", "docB"], ["docA"]) == 1 / 2
    assert mrr(["docA"], ["docA"]) == 1.0
    assert mrr(["docX"], ["docA"]) == 0.0


def test_ndcg_perfect_ranking_is_one():
    ranked = ["docA", "docB", "docC"]
    gold = ["docA", "docB"]
    assert abs(ndcg_at_k(ranked, gold, k=3) - 1.0) < 1e-9


def test_ndcg_worse_ranking_scores_lower():
    gold = ["docA", "docB"]
    best = ndcg_at_k(["docA", "docB", "docC"], gold, k=3)
    worse = ndcg_at_k(["docC", "docA", "docB"], gold, k=3)
    assert worse < best


def test_fuse_hybrid_weights_dense_vs_bm25():
    dense = [("d1", 0.9), ("d2", 0.1)]
    bm25 = [("d2", 10.0), ("d1", 1.0)]
    dense_only = fuse_hybrid(dense, bm25, alpha=1.0, top_k=2)
    assert dense_only[0][0] == "d1"
    bm25_only = fuse_hybrid(dense, bm25, alpha=0.0, top_k=2)
    assert bm25_only[0][0] == "d2"


def test_fuse_hybrid_handles_disjoint_result_sets():
    dense = [("d1", 0.9)]
    bm25 = [("d2", 5.0)]
    fused = fuse_hybrid(dense, bm25, alpha=0.5, top_k=5)
    assert {cid for cid, _ in fused} == {"d1", "d2"}
