import re
from pathlib import Path
from typing import Any

from app.parsers.base import ParsedSection, read_text_file


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownParser:
    """按 Markdown 标题层级输出章节。"""

    def parse(
        self,
        path: Path,
        source: str,
    ) -> list[ParsedSection]:
        lines = read_text_file(path).splitlines()
        front_matter, lines = self._extract_front_matter(lines)

        sections: list[ParsedSection] = []
        heading_stack: list[str] = []
        content_lines: list[str] = []

        def save_section() -> None:
            body = "\n".join(content_lines).strip()

            if not body:
                return

            title_path = list(heading_stack)
            searchable_content = body

            if title_path:
                searchable_content = (
                    f"{' > '.join(title_path)}\n{body}"
                )

            metadata: dict[str, Any] = {
                "document_type": "markdown",
                "title_path": title_path,
                **front_matter,
            }

            metadata.setdefault(
                "doc_id",
                front_matter.get("doc_id", source),
            )

            sections.append(
                ParsedSection(
                    source=source,
                    content=searchable_content,
                    metadata=metadata,
                )
            )

        for line in lines:
            match = HEADING_PATTERN.match(line)

            if match is None:
                content_lines.append(line)
                continue

            save_section()
            content_lines.clear()

            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(title)

        save_section()
        return sections

    def _extract_front_matter(
        self,
        lines: list[str],
    ) -> tuple[dict[str, str], list[str]]:
        """解析简单的 key: value Markdown Front Matter。"""
        if not lines or lines[0].strip() != "---":
            return {}, lines

        closing_index: int | None = None

        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                closing_index = index
                break

        if closing_index is None:
            return {}, lines

        metadata: dict[str, str] = {}

        for line in lines[1:closing_index]:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            cleaned_key = key.strip()
            cleaned_value = value.strip().strip("\"'")

            if cleaned_key:
                metadata[cleaned_key] = cleaned_value

        return metadata, lines[closing_index + 1 :]
