from __future__ import annotations

import os
import re
import math
import json
from typing import List, Dict, Any
import base64
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

import requests
from langchain_core.tools import tool


MAX_BYTES = 400_000  # ~400KB cap for fetched content
# Configurable timeouts (updated via set_timeouts from the app)
WEB_GET_TIMEOUT = 15
WEB_SEARCH_TIMEOUT = 8
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)

# Default OCR (vision) model for Ollama
OLLAMA_OCR_MODEL = os.getenv("OLLAMA_OCR_MODEL", "llava:34b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))  # seconds


def _clean_text(html: str) -> str:
    """Try BeautifulSoup if available; otherwise fall back to regex stripping."""
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ")
    except ImportError:
        # Naive fallback: strip script/style and tags
        import html as htmlmod
        no_script = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        no_style = re.sub(r"<style[\s\S]*?</style>", " ", no_script, flags=re.IGNORECASE)
        no_tags = re.sub(r"<[^>]+>", " ", no_style)
        text = htmlmod.unescape(no_tags)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@tool("web_get", return_direct=False)
def web_get(url: str) -> str:
    """Fetch and extract readable text from the given URL.

    Input: full URL (http/https). Returns truncated text (character limit)
    along with basic HTTP header metadata. Use this when you have
    a specific link to analyze. Does not execute JavaScript.
    """
    if not re.match(r"^https?://", url):
        return "Error: provide absolute http(s) URL"
    try:
        with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=WEB_GET_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            raw = b""
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    raw += chunk
                    if len(raw) > MAX_BYTES:
                        break
            if "html" in content_type.lower():
                text = _clean_text(raw.decode(errors="ignore"))
            else:
                text = raw.decode(errors="ignore")
        if len(text) > 20_000:
            text = text[:20_000] + "\n...[truncated]"
        return json.dumps({
            "url": url,
            "content_type": content_type,
            "text": text,
        })
    except Exception as e:
        return f"Error fetching {url}: {e}"


@tool("web_search", return_direct=False)
def web_search(query: str) -> str:
    """Quick search (DuckDuckGo Instant Answer API).

    Input: text query. Returns a short JSON with title/abstract
    and related links (limited; not full Google results).
    Use when you need a list of potential sources to check.
    """
    try:
        # Prefer robust library if available
        try:
            from duckduckgo_search import DDGS  # type: ignore
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=10):
                    results.append({
                        "title": r.get("title"),
                        "href": r.get("href"),
                        "body": r.get("body"),
                    })
            return json.dumps({"query": query, "results": results})
        except ImportError:
            # Fallback: Instant Answer API (limited)
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
                headers={"User-Agent": USER_AGENT},
                timeout=WEB_SEARCH_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            out: Dict[str, Any] = {
                "Heading": data.get("Heading"),
                "Abstract": data.get("Abstract"),
                "AbstractURL": data.get("AbstractURL"),
                "RelatedTopics": [],
            }
            topics = []
            for t in data.get("RelatedTopics", [])[:10]:
                if isinstance(t, dict) and "Text" in t and "FirstURL" in t:
                    topics.append({"Text": t.get("Text"), "FirstURL": t.get("FirstURL")})
                elif isinstance(t, dict) and "Topics" in t:
                    for tt in t["Topics"][:5]:
                        if isinstance(tt, dict) and "Text" in tt and "FirstURL" in tt:
                            topics.append({"Text": tt.get("Text"), "FirstURL": tt.get("FirstURL")})
            out["RelatedTopics"] = topics[:10]
            return json.dumps(out)
    except Exception as e:
        return f"Error searching: {e}"


def _safe_eval_math(expr: str) -> float:
    """Safe math evaluator using sympy for + - * / ** () and math functions."""
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    # Disallow dangerous characters
    if re.search(r"[^0-9eE\.\+\-\*/\(\)\,\s\^a-zA-Z_]", expr):
        raise ValueError("Unsupported characters in expression")
    expr = expr.replace("^", "**")
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try:
        parsed = parse_expr(expr, transformations=transformations, evaluate=True)
        result = float(parsed.evalf())
    except Exception as e:
        raise ValueError(f"Invalid math expression: {e}")
    return result


@tool("calculator", return_direct=False)
def calculator(expr: str) -> str:
    """Oblicz wyrażenie matematyczne. Obsługuje + - * / ^, nawiasy i funkcje z math (np. sin, log).

    Wejście: np. "2*(3+4)^2" albo "log(100,10)".
    Zwraca wynik jako string.
    """
    try:
        val = _safe_eval_math(expr)
        return str(val)
    except Exception as e:
        return f"Error: {e}"


@tool("now", return_direct=False)
def now(_arg: str = "") -> str:
    """Zwróć bieżącą datę i czas: ISO8601 lokalnie i w UTC."""
    try:
        local = datetime.now()
        utc = datetime.now(timezone.utc)
        return json.dumps({
            "local_iso": local.isoformat(timespec="seconds"),
            "utc_iso": utc.isoformat(timespec="seconds"),
            "date": local.strftime("%Y-%m-%d"),
            "time": local.strftime("%H:%M:%S"),
        })
    except Exception as e:
        return f"Error getting time: {e}"


@tool("today", return_direct=False)
def today(_arg: str = "") -> str:
    """Zwróć dzisiejszą datę w formacie YYYY-MM-DD."""
    try:
        return datetime.now().strftime("%Y-%m-%d")
    except Exception as e:
        return f"Error getting date: {e}"


@tool("read_file", return_direct=False)
def read_file(path: str) -> str:
    """Przeczytaj plik tekstowy z lokalnego katalogu roboczego.

    Wejście: ścieżka relatywna do projektu. Zwraca do 50k znaków.
    Nie używaj do plików binarnych.
    """
    path = os.path.normpath(path)
    if os.path.isabs(path) or path.startswith(".."):
        return "Error: only relative paths within workspace are allowed"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if len(text) > 50_000:
def get_basic_tools():
    """Export a default list of basic tools."""
    return [now, today, time_in, web_search, web_get, calculator, read_file, ocr_image]


@tool("ocr_image", return_direct=False)
def ocr_image(path_or_url: str) -> str:
    """Extract readable text from an image using a local Ollama vision model.

    Input: path (relative to workspace) or http(s) URL to an image.
    Uses model from env `OLLAMA_OCR_MODEL` (default: llava:34b).
    Returns the transcribed text, preserving line breaks when possible.
    """
    spec = (path_or_url or "").strip()
    if not spec:
        return "Error: provide an image path or URL"
    try:
        if re.match(r"^https?://", spec):
            b64 = _b64_from_url(spec, timeout=WEB_GET_TIMEOUT)
        else:
            b64 = _b64_from_local(spec)
    except Exception as e:
        return f"Error loading image: {e}"

    model = OLLAMA_OCR_MODEL
    base_url = OLLAMA_BASE_URL.rstrip("/")
    prompt = (
        "OCR the image. Extract the visible text as plain text. "
        "Preserve reading order and line breaks. Do not add commentary."
    )
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        if not text:
            return "Error: empty OCR response"
        return text
    except Exception as e:
        return f"Error calling Ollama OCR ({model}): {e}"
        return f"Error reading {path}: {e}"


def get_basic_tools():
    """Export a default list of basic tools."""
    return [now, today, time_in, web_search, web_get, calculator, read_file, ocr_image]


def select_tools_for_query(query: str):
    """Return a minimal, relevant subset of tools for the query.
    This reduces confusion for ReAct agents and improves reliability.
    """
    q = (query or "").lower()
    tools = []

    def add(t):
        if t not in tools:
            tools.append(t)

    # Date/time intents
    if any(k in q for k in [
        "today", "current date", "current time", "what time", "timezone", "utc", "now",
        "dzisiaj", "dzisiejsza data", "aktualna data", "bieżąca data", "aktualny czas", "ktora jest godzina", "która jest godzina", "czas w", "strefa czasowa"
    ]):
        # Try IANA tz first
        if ZoneInfo is not None and spec and ("/" in spec or spec.upper() == "UTC"):
            try:
                tzinfo = ZoneInfo(spec if spec else "UTC")
                label = spec
            except (KeyError, ValueError, AttributeError):
                tzinfo = None
        return tools

    # Search intents
    if any(k in q for k in ["search", "find", "google", "bing", "szukaj", "wyszukaj"]):
        add(web_search); add(web_get)
        return tools

    # Math
    if any(op in q for op in ["+", "-", "*", "/", "^"]) or any(k in q for k in ["oblicz", "policz", "sum", "log", "sqrt"]):
        add(calculator)

    # Files
    if any(k in q for k in ["read file", "open file", "plik", "file "]):
        add(read_file)

    # OCR / image intents
    if any(k in q for k in [
        "ocr", "image text", "read image", "screenshot", "extract text from image", "scan", "photo text"
    ]):
        add(ocr_image)

    # Default minimal set
    for t in [web_search, web_get, calculator, now, today]:
        add(t)
    return tools


@tool("time_in", return_direct=False)
def time_in(spec: str) -> str:
    """Zwróć aktualny czas w podanej strefie. Wejście:
    - Nazwa IANA, np. 'Europe/Warsaw', 'America/New_York'
    - Albo offset 'UTC+02:00', 'UTC-5', '+01:30', '-04'
    Zwraca JSON z polami time/date/iso/tz.
    """
    spec = (spec or "").strip()
    try:
        now_utc = datetime.now(timezone.utc)
        tzinfo = None
        label = ""
        # Try IANA tz first
        if ZoneInfo is not None and spec and ("/" in spec or spec.upper() == "UTC"):
            try:
                tzinfo = ZoneInfo(spec if spec else "UTC")
                label = spec
            except Exception:
                tzinfo = None
        if tzinfo is None:
            # Parse offsets like UTC+2, +02:00, -5, etc.
            m = re.fullmatch(r"(?:UTC)?\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?", spec, flags=re.IGNORECASE)
            if m:
                sign, hh, mm = m.group(1), int(m.group(2)), m.group(3)
                minutes = int(mm) if mm else 0
                total = hh * 60 + minutes
                if sign == "-":
                    total = -total
                tzinfo = timezone(timedelta(minutes=total))
                label = f"UTC{sign}{hh:02d}:{minutes:02d}"
            else:
                # default UTC
                tzinfo = timezone.utc
                label = "UTC"
        local = now_utc.astimezone(tzinfo)
        return json.dumps({
            "tz": label,
            "iso": local.isoformat(timespec="seconds"),
            "date": local.strftime("%Y-%m-%d"),
            "time": local.strftime("%H:%M:%S"),
        })
    except Exception as e:
        return f"Error in time_in: {e}"


def set_timeouts(web_get_timeout: int | None = None, web_search_timeout: int | None = None) -> None:
    """Update module-level HTTP timeouts for tools."""
    global WEB_GET_TIMEOUT, WEB_SEARCH_TIMEOUT
    if isinstance(web_get_timeout, int) and web_get_timeout > 0:
        WEB_GET_TIMEOUT = web_get_timeout
    if isinstance(web_search_timeout, int) and web_search_timeout > 0:
        WEB_SEARCH_TIMEOUT = web_search_timeout


def _b64_from_local(path: str) -> str:
    path = os.path.normpath(path)
    if os.path.isabs(path) or path.startswith(".."):
        raise ValueError("only relative paths within workspace are allowed")
    with open(path, "rb") as f:
        data = f.read()
    if len(data) > 8 * 1024 * 1024:
        # Safety cap 8MB for local images
        raise ValueError("image too large (>8MB)")
    return base64.b64encode(data).decode()


def _b64_from_url(url: str, timeout: int) -> str:
    if not re.match(r"^https?://", url):
        raise ValueError("url must be http(s)")
    with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "").lower()
        if not any(x in content_type for x in ["image/", "octet-stream"]):
            # still try to read, but warn via exception so agent can react
            pass
        raw = b""
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                raw += chunk
                if len(raw) > 8 * 1024 * 1024:
                    break
        if not raw:
            raise ValueError("empty image response")
    return base64.b64encode(raw).decode()


@tool("ocr_image", return_direct=False)
def ocr_image(path_or_url: str) -> str:
    """Extract readable text from an image using a local Ollama vision model.

    Input: path (relative to workspace) or http(s) URL to an image.
    Uses model from env `OLLAMA_OCR_MODEL` (default: llava:34b).
    Returns the transcribed text, preserving line breaks when possible.
    """
    spec = (path_or_url or "").strip()
    if not spec:
        return "Error: provide an image path or URL"
    try:
        if re.match(r"^https?://", spec):
            b64 = _b64_from_url(spec, timeout=WEB_GET_TIMEOUT)
        else:
            b64 = _b64_from_local(spec)
    except Exception as e:
        return f"Error loading image: {e}"

    model = OLLAMA_OCR_MODEL
    base_url = OLLAMA_BASE_URL.rstrip("/")
    prompt = (
        "OCR the image. Extract the visible text as plain text. "
        "Preserve reading order and line breaks. Do not add commentary."
    )
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        if not text:
            return "Error: empty OCR response"
        return text
    except Exception as e:
        return f"Error calling Ollama OCR ({model}): {e}"
