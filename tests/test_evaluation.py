from tests.evaluate import (
    find_missing_keyword_requirements,
)


def test_keyword_requirement_accepts_synonym_group() -> None:
    missing = find_missing_keyword_requirements(
        answer="订单发货后无法修改收货地址。",
        requirements=[
            ["不能修改", "无法修改", "不可以修改"],
        ],
    )

    assert missing == []


def test_keyword_requirement_still_requires_every_concept() -> None:
    missing = find_missing_keyword_requirements(
        answer="产品进水不在保修范围内。",
        requirements=[
            "进水",
            ["免费保修", "免费维修"],
        ],
    )

    assert missing == ["免费保修/免费维修"]
