"""WebFetch and WebSearch.

WebFetch returns the page as text; no second model summarises it, so ``prompt``
only tells the reader of the transcript why the page was fetched. Addresses on
the loopback or a private network are refused unless the settings allow them: a
fetched page can carry instructions, and the local services have no login.

WebSearch needs no account: it reads DuckDuckGo's plain HTML result page. With
``BRAVE_API_KEY`` set it uses the Brave search API instead.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx

from mwm_harness.tools.base import Tool, ToolContext, ToolResult

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) mwm-harness/0.1"
MAX_BYTES = 5_000_000
MAX_REDIRECTS = 5
FETCH_TIMEOUT = 30.0
SKIPPED_TAGS = {"script", "style", "noscript", "svg", "template", "iframe"}
BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "main", "aside", "nav", "br", "tr",
    "table", "ul", "ol", "pre", "blockquote", "form", "figure", "hr", "dl", "dt", "dd",
}  # fmt: skip
HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "##### ", "h6": "###### "}


class PageText(HTMLParser):
    """HTML to readable text: headings, list items and link targets survive."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False
        self._pre = 0
        self._links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in SKIPPED_TAGS:
            self._skip += 1
        if self._skip:
            return
        if tag in HEADINGS:
            self.parts.append("\n\n" + HEADINGS[tag])
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "pre":
            self._pre += 1
        if tag == "a":
            href = dict(attrs).get("href") or ""
            keep = href and not href.startswith(("#", "javascript:", "mailto:"))
            self._links.append(urljoin(self.base_url, href) if keep else "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in SKIPPED_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "pre":
            self._pre = max(0, self._pre - 1)
        if tag == "a" and self._links:
            href = self._links.pop()
            if href:
                self.parts.append(f" ({href})")
        if tag in HEADINGS or tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip:
            return
        self.parts.append(data if self._pre else re.sub(r"\s+", " ", data))

    def text(self) -> str:
        joined = "".join(self.parts)
        joined = re.sub(r"[ \t]+\n", "\n", joined)
        joined = re.sub(r"\n[ \t]+", "\n", joined)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def html_to_text(html: str, base_url: str) -> tuple[str, str]:
    parser = PageText(base_url)
    parser.feed(html)
    parser.close()
    return parser.title.strip(), parser.text()


async def private_address(host: str) -> bool:
    """True when the host is, or resolves to, a loopback, private or link-local address."""
    try:
        addresses = [ipaddress.ip_address(host.strip("[]"))]
    except ValueError:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                host, None, type=socket.SOCK_STREAM
            )
        except OSError:
            return False  # does not resolve: the request itself will report that
        addresses = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
    return any(
        a.is_private or a.is_loopback or a.is_link_local or a.is_reserved or a.is_unspecified
        for a in addresses
    )


class WebFetch(Tool):
    name = "WebFetch"
    description = (
        "Fetch a URL and return its content as text (HTML is converted, JSON and plain text "
        "are returned as they are). prompt states what you are looking for on the page. "
        "Local and private network addresses are refused."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "prompt": {"type": "string", "description": "What to look for on the page"},
        },
        "required": ["url"],
    }

    def __init__(
        self, allow_private: bool = False, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.allow_private = allow_private
        self.transport = transport

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = str(tool_input["url"]).strip()
        if "://" not in url:
            url = "https://" + url
        async with httpx.AsyncClient(
            transport=self.transport, headers={"User-Agent": USER_AGENT}, timeout=FETCH_TIMEOUT
        ) as client:
            try:
                for _ in range(MAX_REDIRECTS + 1):
                    parts = urlsplit(url)
                    if parts.scheme not in ("http", "https") or not parts.hostname:
                        return ToolResult(f"not an http(s) URL: {url}", True)
                    if not self.allow_private and await private_address(parts.hostname):
                        return ToolResult(
                            f"refused: {parts.hostname} is a local or private address "
                            "(web_allow_private in settings.toml lifts this)",
                            True,
                        )
                    async with client.stream("GET", url) as response:
                        if response.is_redirect and response.headers.get("location"):
                            url = urljoin(url, response.headers["location"])
                            continue
                        body = bytearray()
                        async for piece in response.aiter_bytes():
                            body.extend(piece)
                            if len(body) > MAX_BYTES:
                                break
                        return self._render(url, response, bytes(body[:MAX_BYTES]), ctx)
                return ToolResult(f"more than {MAX_REDIRECTS} redirects, last: {url}", True)
            except httpx.HTTPError as exc:
                return ToolResult(f"fetch failed: {type(exc).__name__}: {exc}", True)

    def _render(
        self, url: str, response: httpx.Response, body: bytes, ctx: ToolContext
    ) -> ToolResult:
        kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
        failed = response.status_code >= 400
        head = f"URL: {url}\nStatus: {response.status_code}\nContent-Type: {kind or 'unknown'}\n"
        textual = (
            kind.startswith("text/") or kind.endswith(("json", "xml", "javascript")) or not kind
        )
        if not textual:
            return ToolResult(f"{head}\n[{len(body):,} bytes of {kind}; not shown as text]", failed)
        text = body.decode(response.encoding or "utf-8", errors="replace")
        if "html" in kind or (not kind and "<html" in text[:2000].lower()):
            title, text = html_to_text(text, url)
            if title:
                head += f"Title: {title}\n"
        return ToolResult(ctx.cap(f"{head}\n{text}"), failed)


class WebSearch(Tool):
    name = "WebSearch"
    description = (
        "Search the web. Returns a numbered list of results with title, URL and snippet. "
        "Follow up with WebFetch on a result to read it."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "description": "1 to 20, default 8"},
        },
        "required": ["query"],
    }
    read_only = True

    def __init__(
        self, brave_key: str = "", transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.brave_key = brave_key or os.environ.get("BRAVE_API_KEY", "")
        self.transport = transport

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(tool_input["query"]).strip()
        if not query:
            return ToolResult("empty query", True)
        limit = max(1, min(int(tool_input.get("max_results") or 8), 20))
        async with httpx.AsyncClient(
            transport=self.transport,
            headers={"User-Agent": USER_AGENT},
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
        ) as client:
            try:
                if self.brave_key:
                    results = await self._brave(client, query, limit)
                else:
                    results = await self._duckduckgo(client, query)
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                return ToolResult(f"search failed: {type(exc).__name__}: {exc}", True)
        if not results:
            return ToolResult(f"no results for: {query}")
        rows = [
            f"{number}. {item['title']}\n   {item['url']}\n   {item['snippet']}".rstrip()
            for number, item in enumerate(results[:limit], 1)
        ]
        return ToolResult(ctx.cap("\n".join(rows)))

    async def _brave(self, client: httpx.AsyncClient, query: str, limit: int) -> list[dict]:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": limit},
            headers={"X-Subscription-Token": self.brave_key, "Accept": "application/json"},
        )
        response.raise_for_status()
        items = (response.json().get("web") or {}).get("results") or []
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": re.sub(r"<[^>]+>", "", item.get("description", "")),
            }
            for item in items
        ]

    async def _duckduckgo(self, client: httpx.AsyncClient, query: str) -> list[dict]:
        response = await client.post("https://html.duckduckgo.com/html/", data={"q": query})
        response.raise_for_status()
        parser = DuckResults()
        parser.feed(response.text)
        return parser.results


class DuckResults(HTMLParser):
    """Pulls ``result__a`` links and ``result__snippet`` texts out of the HTML result page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._field = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if "result__a" in classes:
            self.results.append(
                {"title": "", "url": real_url(attributes.get("href") or ""), "snippet": ""}
            )
            self._field = "title"
        elif "result__snippet" in classes and self.results:
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._field = ""

    def handle_data(self, data: str) -> None:
        if self._field and self.results:
            self.results[-1][self._field] += data


def real_url(href: str) -> str:
    """DuckDuckGo wraps each result in a redirect link; ``uddg`` holds the target."""
    if href.startswith("//"):
        href = "https:" + href
    target = parse_qs(urlsplit(href).query).get("uddg")
    return target[0] if target else href
