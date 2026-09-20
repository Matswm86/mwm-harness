"""WebFetch and WebSearch, offline through a mocked transport."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from mwm_harness.sandbox import Sandbox
from mwm_harness.tools.base import ToolContext
from mwm_harness.tools.web import WebFetch, WebSearch, html_to_text, real_url

PUBLIC = "http://93.184.216.34"

PAGE = """<html><head><title>A  page</title><style>p {color: red}</style></head>
<body><script>alert(1)</script><h1>Heading</h1><p>First   paragraph with a
<a href="/docs">link</a>.</p><ul><li>one</li><li>two</li></ul>
<pre>keep   spacing</pre></body></html>"""

DUCK = """<div class="result"><a rel="nofollow" class="result__a"
href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&amp;rut=x">First <b>hit</b></a>
<a class="result__snippet" href="x">Snippet <b>one</b>.</a></div>
<div class="result"><a class="result__a" href="https://example.org/b">Second</a>
<a class="result__snippet" href="x">Snippet two.</a></div>"""


def run(coro):
    return asyncio.run(coro)


def context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, scratch=tmp_path / "scratch", sandbox=Sandbox("off", []))


def test_html_to_text_keeps_structure_and_drops_scripts():
    title, text = html_to_text(PAGE, "https://example.org/x/")
    assert title == "A  page"
    assert "# Heading" in text
    assert "First paragraph with a link (https://example.org/docs)." in text
    assert "- one\n- two" in text
    assert "keep   spacing" in text
    assert "alert" not in text and "color" not in text


def test_fetch_follows_a_redirect_and_converts_html(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"Location": "/new"})
        return httpx.Response(200, text=PAGE, headers={"Content-Type": "text/html; charset=utf-8"})

    tool = WebFetch(transport=httpx.MockTransport(handler))
    result = run(tool.run({"url": f"{PUBLIC}/old"}, context(tmp_path)))
    assert not result.is_error
    assert f"URL: {PUBLIC}/new" in result.content
    assert "Title: A  page" in result.content and "# Heading" in result.content


def test_fetch_refuses_loopback_and_a_redirect_into_a_private_network(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://10.0.0.5/admin"})

    tool = WebFetch(transport=httpx.MockTransport(handler))
    direct = run(tool.run({"url": "http://127.0.0.1:8088/mcp"}, context(tmp_path)))
    assert direct.is_error and "private address" in direct.content
    hopped = run(tool.run({"url": f"{PUBLIC}/go"}, context(tmp_path)))
    assert hopped.is_error and "10.0.0.5" in hopped.content


def test_fetch_allows_private_when_the_setting_says_so(tmp_path):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    tool = WebFetch(allow_private=True, transport=transport)
    result = run(tool.run({"url": "http://127.0.0.1:9/x"}, context(tmp_path)))
    assert not result.is_error and '"ok"' in result.content


def test_fetch_reports_http_errors_and_binary_bodies(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/missing":
            return httpx.Response(404, text="nope", headers={"Content-Type": "text/plain"})
        return httpx.Response(200, content=b"\x89PNG....", headers={"Content-Type": "image/png"})

    tool = WebFetch(transport=httpx.MockTransport(handler))
    missing = run(tool.run({"url": f"{PUBLIC}/missing"}, context(tmp_path)))
    assert missing.is_error and "Status: 404" in missing.content
    image = run(tool.run({"url": f"{PUBLIC}/pic"}, context(tmp_path)))
    assert "not shown as text" in image.content
    assert run(tool.run({"url": "ftp://x/y"}, context(tmp_path))).is_error


def test_search_parses_duckduckgo_results(tmp_path):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=DUCK))
    tool = WebSearch(transport=transport)
    result = run(tool.run({"query": "anything", "max_results": 1}, context(tmp_path)))
    assert result.content == "1. First hit\n   https://example.org/a\n   Snippet one."
    both = run(tool.run({"query": "anything"}, context(tmp_path)))
    assert "2. Second\n   https://example.org/b" in both.content


def test_search_uses_brave_when_a_key_is_set(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Subscription-Token"] == "k"
        item = {"title": "T", "url": "https://t.example", "description": "a <strong>b</strong>"}
        return httpx.Response(200, json={"web": {"results": [item]}})

    tool = WebSearch(brave_key="k", transport=httpx.MockTransport(handler))
    result = run(tool.run({"query": "q"}, context(tmp_path)))
    assert result.content == "1. T\n   https://t.example\n   a b"


def test_real_url_unwraps_the_redirect():
    assert real_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.b%2Fc") == "https://a.b/c"
    assert real_url("https://plain.example/") == "https://plain.example/"
