import ast
from pathlib import Path

from app.parsers.base import ParsedSection, read_text_file


class PythonCodeParser:
    """按照Python顶层类和函数边界解析代码。"""

    def parse(
        self,
        path: Path,
        source: str,
    ) -> list[ParsedSection]:
        source_code = read_text_file(path)

        try:
            tree = ast.parse(source_code, filename=str(path))
        except SyntaxError:
            # 未完成或存在语法错误的代码也应该能进入知识库，
            # 此时降级为按行解析，并保留行号。
            return GenericCodeParser("python").parse(
                path=path,
                source=source,
            )

        lines = source_code.splitlines()
        sections: list[ParsedSection] = []
        cursor = 1

        def save_module_section(
            start_line: int,
            end_line: int,
        ) -> None:
            if end_line < start_line:
                return

            content = "\n".join(
                lines[start_line - 1 : end_line]
            ).strip()

            if not content:
                return

            sections.append(
                ParsedSection(
                    source=source,
                    content=content,
                    metadata={
                        "document_type": "code",
                        "language": "python",
                        "symbol_type": "module",
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                )
            )

        for node in tree.body:
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                continue

            decorator_lines = [
                decorator.lineno
                for decorator in getattr(
                    node,
                    "decorator_list",
                    [],
                )
            ]
            start_line = min(
                [node.lineno, *decorator_lines]
            )
            end_line = node.end_lineno or node.lineno

            save_module_section(
                start_line=cursor,
                end_line=start_line - 1,
            )

            content = "\n".join(
                lines[start_line - 1 : end_line]
            ).strip()

            if not content:
                continue

            symbol_type = (
                "class"
                if isinstance(node, ast.ClassDef)
                else "function"
            )

            sections.append(
                ParsedSection(
                    source=source,
                    content=content,
                    metadata={
                        "document_type": "code",
                        "language": "python",
                        "symbol": node.name,
                        "symbol_type": symbol_type,
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                )
            )
            cursor = end_line + 1

        save_module_section(
            start_line=cursor,
            end_line=len(lines),
        )

        return sections


class GenericCodeParser:
    """其他代码文件按行窗口切分，并保留行号。"""

    def __init__(
        self,
        language: str,
        max_lines: int = 120,
        overlap_lines: int = 20,
    ) -> None:
        if max_lines <= 0:
            raise ValueError("max_lines必须大于0")
        if overlap_lines < 0 or overlap_lines >= max_lines:
            raise ValueError(
                "overlap_lines必须大于等于0且小于max_lines"
            )

        self.language = language
        self.max_lines = max_lines
        self.overlap_lines = overlap_lines

    def parse(
        self,
        path: Path,
        source: str,
    ) -> list[ParsedSection]:
        lines = read_text_file(path).splitlines()

        if not any(line.strip() for line in lines):
            return []

        sections: list[ParsedSection] = []
        start = 0
        step = self.max_lines - self.overlap_lines

        while start < len(lines):
            end = min(start + self.max_lines, len(lines))
            content = "\n".join(lines[start:end]).strip()

            if content:
                sections.append(
                    ParsedSection(
                        source=source,
                        content=content,
                        metadata={
                            "document_type": "code",
                            "language": self.language,
                            "start_line": start + 1,
                            "end_line": end,
                        },
                    )
                )

            if end >= len(lines):
                break

            start += step

        return sections
