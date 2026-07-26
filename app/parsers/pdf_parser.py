from pathlib import Path

from app.parsers.base import ParsedSection


class PdfParser:
    """按页解析带文本层的 PDF。"""

    def parse(
        self,
        path: Path,
        source: str,
    ) -> list[ParsedSection]:
        try:
            import pymupdf
        except ImportError as error:
            raise RuntimeError(
                "解析PDF需要安装PyMuPDF"
            ) from error

        sections: list[ParsedSection] = []

        with pymupdf.open(path) as document:
            total_pages = document.page_count

            for page_number, page in enumerate(
                document,
                start=1,
            ):
                content = page.get_text(
                    "text",
                    sort=True,
                ).strip()

                if not content:
                    image_count = len(
                        page.get_images(full=True)
                    )
                    parse_status = (
                        "ocr_required"
                        if image_count > 0
                        else "empty_page"
                    )

                    sections.append(
                        ParsedSection(
                            source=source,
                            content="",
                            metadata={
                                "document_type": "pdf",
                                "page": page_number,
                                "total_pages": total_pages,
                                "image_count": image_count,
                                "parse_status": parse_status,
                            },
                        )
                    )
                    continue

                sections.append(
                    ParsedSection(
                        source=source,
                        content=content,
                        metadata={
                            "document_type": "pdf",
                            "page": page_number,
                            "total_pages": total_pages,
                            "parse_status": "parsed",
                        },
                    )
                )

        return sections
