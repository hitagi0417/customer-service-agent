from app.bm25 import BM25Index, tokenize_for_search


def test_search_tokenizer_supports_chinese_and_product_codes() -> None:
    tokens = tokenize_for_search(
        "退款时间 XJ-900 order_123"
    )

    assert "退款" in tokens
    assert "款时" in tokens
    assert "时间" in tokens
    assert "xj-900" in tokens
    assert "order_123" in tokens


def test_bm25_prefers_exact_keyword_document() -> None:
    index = BM25Index(
        [
            "XJ-900型号产品支持两年保修。",
            "普通商品支持七天退款。",
        ]
    )

    scores = index.normalized_scores(
        "XJ-900保修多久"
    )

    assert scores[0] == 1.0
    assert scores[1] == 0.0
