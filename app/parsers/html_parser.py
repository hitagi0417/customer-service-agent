from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from app.parsers.base import ParsedSection, read_text_file


class HtmlParser:
    """解析本地HTML或已经安全下载的网页内容。"""

    def parse(
        self,
        path: Path,
        source: str,
    ) -> list[ParsedSection]:
        html = read_text_file(path)
        return self.parse_html(
            html=html,
            source=source,
            source_url=None,
        )

    def parse_html(
        self,
        html: str,
        source: str,
        source_url: str | None,
    ) -> list[ParsedSection]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as error:
            raise RuntimeError(
                "解析HTML需要安装beautifulsoup4"
            ) from error

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup.select(
            "script, style, noscript, nav, footer, header, aside"
        ):
            tag.decompose()

        title = ""

        if soup.title:
            title = soup.title.get_text(" ", strip=True)

        root = (
            soup.select_one("main")
            or soup.select_one("article")
            or soup.body
            or soup
        )

        content = root.get_text("\n", strip=True)

        if not content:
            return []

        metadata = {
            "document_type": "html",
            "title": title,
        }

        if source_url:
            metadata.update(
                {
                    "url": source_url,
                    "fetched_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
            )

        return [
            ParsedSection(
                source=source,
                content=content,
                metadata=metadata,
            )
        ]


class WebFetcher:
    """只抓取管理员允许域名中的HTTPS网页。"""

    def __init__(
        self,
        allowed_domains: set[str],
        timeout_seconds: float = 10.0,
        max_content_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        self.allowed_domains = {
            domain.strip().lower()
            for domain in allowed_domains
            if domain.strip()
        }
        self.timeout_seconds = timeout_seconds
        self.max_content_bytes = max_content_bytes

    def fetch(self, url: str) -> str:
        try:
            import httpx
        except ImportError as error:
            raise RuntimeError(
                "抓取网页需要安装httpx"
            ) from error

        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        if parsed.scheme != "https":
            raise ValueError("只允许抓取HTTPS网页")

        if parsed.username or parsed.password:
            raise ValueError("网页URL不能包含认证信息")

        if hostname not in self.allowed_domains:
            raise ValueError(
                f"网页域名不在允许列表中：{hostname}"
            )

        response = httpx.get(
            url,
            timeout=self.timeout_seconds,
            follow_redirects=False,
            headers={
                "User-Agent": "CustomerServiceKnowledgeBot/1.0",
            },
        )

        if response.is_redirect:
            raise ValueError("网页发生重定向，已拒绝抓取")

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type",
            "",
        ).lower()

        if "text/html" not in content_type:
            raise ValueError("URL返回内容不是HTML")

        if len(response.content) > self.max_content_bytes:
            raise ValueError("网页内容超过大小限制")

        return response.text
