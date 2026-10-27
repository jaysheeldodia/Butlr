"""Visible-text cleaning for email bodies before they reach an LLM.

Shared by the labeler (scripts/label_runner.py), the extractor
(agents/extractor.py), and the benchmark (scripts/benchmark.py) so that
labeling, training, and serving all see the SAME cleaned text. Without this,
an HTML-heavy body truncated at the 4000-char cap becomes a wall of CSS and
the model (and the human) cannot read the message.

The logic is the audit-proven version from scripts/audit_empties.py:
quoted lines cut, <style>/<script>/... blocks removed (even when truncated),
HTML parsed to visible text, entities decoded, CSS removed by signature
density (works for minified or bare CSS without tags), long no-space blobs
dropped.

If no readable text remains, clean_body() returns "" - callers must treat
"" as "return []".
"""
from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser

QUOTE_RE = re.compile(r"^[ \t]*>.*(?:\n|\Z)", re.MULTILINE)
SKIP_BLOCK_RE = re.compile(
    r"(?is)<(style|script|head|title|noscript)\b[^>]*>.*?(?:</\1\s*>|\Z)")
# Structural CSS lines that can sit under the density threshold when the CSS
# is pretty-printed: at-rules, selector/brace lines, `prop: value;`.
CSS_LINE_RE = re.compile(
    r"^@(?:media|font-face|import|keyframes|charset|supports|page|-[\w-]+)\b"
    r"|[{}]\s*$"
    r"|^-?[a-z][a-z-]*\s*:\s*[^;]+;\s*$")
# `selector { ... }` on one rule; an unclosed block runs to end of input.
CSS_RULE_RE = re.compile(r"[^{}\n]*\{[^{}]*(?:\}|\Z)")
CSS_DENSITY_MAX = 0.05
DENSITY_MIN_LEN = 40  # shorter lines ("Thanks!") are judged by CSS_LINE_RE only
BLOCK_TAGS = {"p", "div", "br", "tr", "li", "td", "h1", "h2", "h3", "table"}


class _TextExtractor(HTMLParser):
    """Extract visible text; skip style/script/head/title/noscript content.

    Guards against unclosed <style>/<script> blocks (common when a body was
    truncated by the 4000-char labeler cap): after 3000 skipped chars we
    stop skipping, so content after a cut-off tag is not silently lost.
    """

    SKIP = {"style", "script", "head", "title", "template", "noscript"}
    SKIP_ESCAPE = 3000

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.skip_chars = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
            self.skip_chars = 0
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            self.skip_chars += len(data)
            if self.skip_chars > self.SKIP_ESCAPE:
                self.skip_depth = 0  # unbalanced block; un-stick the parser
                self.parts.append(data)
            return
        self.parts.append(data)


def _css_density(s: str) -> float:
    """Share of CSS signature chars ({, }, ;, !, @) in a string."""
    if not s:
        return 0.0
    sig = sum(1 for ch in s if ch in "{};!@")
    return sig / len(s)


def _is_css_like(s: str) -> bool:
    if CSS_LINE_RE.search(s):
        return True
    return len(s) >= DENSITY_MIN_LEN and _css_density(s) > CSS_DENSITY_MAX


def clean_body(body: str, max_chars: int = 900) -> str:
    """Return the readable text of an email body, or "" if none survives.

    CSS (even bare text without <style> tags, common after truncation) is
    removed by signature density, so a body that lost its message to the
    4000-char cap returns "" instead of a CSS wall.
    """
    text = QUOTE_RE.sub("", body or "")
    # Closing tag may be cut off by truncation: \Z matches to end of input.
    text = SKIP_BLOCK_RE.sub("\n", text)
    try:
        parser = _TextExtractor()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)  # convert_charrefs=True decodes entities
    except Exception:
        text = unescape(re.sub(r"<[^>]*>", "\n", text))
    # Innermost rules go first, so loop until nested @media blocks are gone.
    prev = None
    while prev != text:
        prev, text = text, CSS_RULE_RE.sub("\n", text)
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if _is_css_like(ln):
            continue
        # long no-space blobs (base64, URLs)
        if len(ln) > 200 and " " not in ln:
            continue
        lines.append(ln)
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not text or _is_css_like(text):
        return ""
    return text[:max_chars]


def clean_body_full(body: str) -> str:
    """Clean without truncation; the caller applies its own cap."""
    return clean_body(body, max_chars=10**9)


def is_html_heavy(body: str) -> bool:
    low = (body or "").lower()
    return bool(re.search(r"<style|@media|class=", low)) or "@media only screen" in low
