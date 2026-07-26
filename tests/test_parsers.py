import base64
import json
from datetime import date
from pathlib import Path

import pymupdf
import pytest

from app.knowledge import KnowledgeLoader
from app.parsers import (
    HtmlParser,
    MarkdownParser,
    ParserRegistry,
    PythonCodeParser,
    WebFetcher,
)


def test_markdown_parser_preserves_heading_path_and_front_matter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "refund.md"
    path.write_text(
        "\n".join(
            (
                "---",
                "doc_id: refund-policy",
                "department: service",
                "---",
                "# 售后政策",
                "退款申请需提供订单号。",
                "## 到账时间",
                "审核通过后原路退回。",
            )
        ),
        encoding="utf-8",
    )

    sections = MarkdownParser().parse(
        path=path,
        source="refund.md",
    )

    assert len(sections) == 2
    assert sections[0].metadata["doc_id"] == "refund-policy"
    assert sections[0].metadata["department"] == "service"
    assert sections[0].metadata["title_path"] == ["售后政策"]
    assert sections[1].metadata["title_path"] == [
        "售后政策",
        "到账时间",
    ]
    assert "售后政策 > 到账时间" in sections[1].content


def test_html_parser_removes_noise_and_keeps_title(
    tmp_path: Path,
) -> None:
    path = tmp_path / "faq.html"
    path.write_text(
        """
        <html>
          <head><title>客服 FAQ</title></head>
          <body>
            <nav>导航噪声</nav>
            <main><h1>退款</h1><p>支持七天退款。</p></main>
            <script>dangerous()</script>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    section = HtmlParser().parse(
        path=path,
        source="faq.html",
    )[0]

    assert section.metadata["title"] == "客服 FAQ"
    assert "支持七天退款" in section.content
    assert "导航噪声" not in section.content
    assert "dangerous" not in section.content


def test_python_parser_preserves_symbols_and_line_numbers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "service.py"
    path.write_text(
        "\n".join(
            (
                "TIMEOUT = 10",
                "",
                "@staticmethod",
                "def query_order(order_id: str):",
                "    return order_id",
                "",
                "class RefundService:",
                "    def create(self):",
                "        return True",
            )
        ),
        encoding="utf-8",
    )

    sections = PythonCodeParser().parse(
        path=path,
        source="service.py",
    )

    module_section = next(
        item
        for item in sections
        if item.metadata["symbol_type"] == "module"
    )
    function_section = next(
        item
        for item in sections
        if item.metadata.get("symbol") == "query_order"
    )
    class_section = next(
        item
        for item in sections
        if item.metadata.get("symbol") == "RefundService"
    )

    assert "TIMEOUT = 10" in module_section.content
    assert function_section.metadata["start_line"] == 3
    assert function_section.metadata["end_line"] == 5
    assert class_section.metadata["symbol_type"] == "class"


def test_invalid_python_falls_back_to_line_parser(
    tmp_path: Path,
) -> None:
    path = tmp_path / "draft.py"
    path.write_text(
        "def unfinished(\n    return 1\n",
        encoding="utf-8",
    )

    section = PythonCodeParser().parse(
        path=path,
        source="draft.py",
    )[0]

    assert section.metadata["language"] == "python"
    assert section.metadata["start_line"] == 1
    assert "unfinished" in section.content


def test_pdf_parser_preserves_page_number(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manual.pdf"

    with pymupdf.open() as document:
        first_page = document.new_page()
        first_page.insert_text((72, 72), "Refund policy")
        second_page = document.new_page()
        second_page.insert_text((72, 72), "Service hours")
        document.save(path)

    sections = ParserRegistry().parse(
        path=path,
        source="manual.pdf",
    )

    assert [item.metadata["page"] for item in sections] == [1, 2]
    assert sections[0].metadata["total_pages"] == 2
    assert "Refund policy" in sections[0].content


def test_registry_and_loader_support_all_requested_formats(
    tmp_path: Path,
) -> None:
    (tmp_path / "policy.md").write_text(
        "# 退款\n七天内可申请。",
        encoding="utf-8",
    )
    (tmp_path / "faq.html").write_text(
        "<main>工作时间为9点到18点。</main>",
        encoding="utf-8",
    )
    (tmp_path / "tool.py").write_text(
        "def query_order():\n    return True\n",
        encoding="utf-8",
    )

    pdf_path = tmp_path / "manual.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Customer service manual")
        document.save(pdf_path)

    chunks = KnowledgeLoader(
        knowledge_dir=tmp_path,
        chunk_size=1000,
        chunk_overlap=100,
    ).load()

    document_types = {
        chunk.metadata["document_type"]
        for chunk in chunks
    }

    assert document_types == {
        "markdown",
        "html",
        "code",
        "pdf",
    }
    assert all(chunk.metadata["content_hash"] for chunk in chunks)
    assert all(chunk.metadata["doc_id"] for chunk in chunks)


@pytest.mark.parametrize(
    ("url", "message"),
    (
        (
            "http://docs.example.com/faq",
            "只允许抓取HTTPS网页",
        ),
        (
            "https://evil.example/faq",
            "网页域名不在允许列表中",
        ),
    ),
)
def test_web_fetcher_rejects_unsafe_urls(
    url: str,
    message: str,
) -> None:
    fetcher = WebFetcher(
        allowed_domains={"docs.example.com"}
    )

    with pytest.raises(ValueError, match=message):
        fetcher.fetch(url)


def test_broken_file_is_isolated_without_blocking_good_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "broken.pdf").write_bytes(
        b"%PDF-1.7\nbroken content"
    )
    (tmp_path / "good.txt").write_text(
        "客服工作时间是周一到周五九点到十八点。",
        encoding="utf-8",
    )

    loader = KnowledgeLoader(
        knowledge_dir=tmp_path
    )
    chunks = loader.load()

    assert len(chunks) == 1
    assert chunks[0].source == "good.txt"
    assert loader.report.failed_files == 1
    assert loader.report.parsed_files == 1
    assert any(
        issue.source == "broken.pdf"
        and issue.reason == "parser_error"
        for issue in loader.report.issues
    )


def test_quality_filter_and_deduplication_are_reported(
    tmp_path: Path,
) -> None:
    valid_content = "退款申请需要在购买后七天内提交。"
    (tmp_path / "policy-a.txt").write_text(
        valid_content,
        encoding="utf-8",
    )
    (tmp_path / "policy-b.txt").write_text(
        valid_content + "\n",
        encoding="utf-8",
    )
    (tmp_path / "policy-copy.txt").write_text(
        valid_content,
        encoding="utf-8",
    )
    (tmp_path / "noise.txt").write_text(
        "@@@",
        encoding="utf-8",
    )

    loader = KnowledgeLoader(
        knowledge_dir=tmp_path,
        min_content_chars=10,
    )
    chunks = loader.load()

    assert len(chunks) == 1
    assert loader.report.duplicate_files == 1
    assert loader.report.duplicate_chunks == 1
    assert loader.report.filtered_sections == 1
    assert any(
        issue.reason == "content_too_short"
        for issue in loader.report.issues
    )


def test_image_only_pdf_is_marked_for_ocr(
    tmp_path: Path,
) -> None:
    # 1×1 PNG，用来模拟只有扫描图片、没有文本层的PDF页面。
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
        "CAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
        "AQUBAScY42YAAAAASUVORK5CYII="
    )
    pdf_path = tmp_path / "scan.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_image(
            pymupdf.Rect(72, 72, 172, 172),
            stream=png_bytes,
        )
        document.save(pdf_path)

    (tmp_path / "good.txt").write_text(
        "这是用于保证知识库仍有有效内容的客服政策。",
        encoding="utf-8",
    )

    loader = KnowledgeLoader(
        knowledge_dir=tmp_path
    )
    chunks = loader.load()

    assert any(
        chunk.source == "good.txt"
        for chunk in chunks
    )
    assert loader.report.ocr_required_pages == 1
    assert any(
        issue.source == "scan.pdf"
        and issue.reason == "ocr_required"
        and issue.metadata["page"] == 1
        for issue in loader.report.issues
    )


def test_ingestion_report_can_be_saved_as_json(
    tmp_path: Path,
) -> None:
    (tmp_path / "policy.txt").write_text(
        "退款申请需要在购买后七天内提交。",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    loader = KnowledgeLoader(
        knowledge_dir=tmp_path
    )

    loader.load()
    loader.save_report(report_path)
    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert report["total_files"] == 1
    assert report["parsed_files"] == 1
    assert report["total_chunks"] == 1
    assert report["completed_at"]


def test_document_status_date_and_version_control(
    tmp_path: Path,
) -> None:
    documents = {
        "policy-v1.md": (
            "---\n"
            "doc_id: refund-policy\n"
            "version: 1.0\n"
            "status: active\n"
            "effective_date: 2025-01-01\n"
            "---\n# 旧政策\n退款期限是5天。"
        ),
        "policy-v2.md": (
            "---\n"
            "doc_id: refund-policy\n"
            "version: 2.0\n"
            "status: active\n"
            "effective_date: 2026-01-01\n"
            "---\n# 当前政策\n退款期限是7天。"
        ),
        "policy-v3-draft.md": (
            "---\n"
            "doc_id: refund-policy\n"
            "version: 3.0\n"
            "status: draft\n"
            "---\n# 草稿政策\n退款期限是14天。"
        ),
        "policy-v4-future.md": (
            "---\n"
            "doc_id: refund-policy\n"
            "version: 4.0\n"
            "status: active\n"
            "effective_date: 2027-01-01\n"
            "---\n# 未来政策\n退款期限是30天。"
        ),
    }

    for filename, content in documents.items():
        (tmp_path / filename).write_text(
            content,
            encoding="utf-8",
        )

    loader = KnowledgeLoader(
        knowledge_dir=tmp_path,
        as_of_date=date(2026, 7, 26),
    )
    chunks = loader.load()

    assert len(chunks) == 1
    assert chunks[0].source == "policy-v2.md"
    assert "7天" in chunks[0].content
    assert loader.report.inactive_documents == 1
    assert loader.report.future_documents == 1
    assert loader.report.superseded_documents == 1
    assert loader.report.skipped_files == 3


def test_invalid_document_metadata_is_quarantined(
    tmp_path: Path,
) -> None:
    (tmp_path / "good.md").write_text(
        "# 有效政策\n当前有效内容。",
        encoding="utf-8",
    )
    (tmp_path / "bad-version.md").write_text(
        (
            "---\nversion: v-next\n---\n"
            "# 错误版本\n不应进入索引。"
        ),
        encoding="utf-8",
    )
    (tmp_path / "bad-date.md").write_text(
        (
            "---\neffective_date: 2026/01/01\n---\n"
            "# 错误日期\n不应进入索引。"
        ),
        encoding="utf-8",
    )

    loader = KnowledgeLoader(
        knowledge_dir=tmp_path,
        as_of_date=date(2026, 7, 26),
    )
    chunks = loader.load()

    assert len(chunks) == 1
    assert chunks[0].source == "good.md"
    assert (
        loader.report.invalid_metadata_documents
        == 2
    )
