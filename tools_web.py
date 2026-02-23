"""
Web Search & Fetch — Production Implementations
------------------------------------------------
Replaces the fragile DuckDuckGo scrape in tools.py.

Search backends (priority order, auto-fallback):
  1. SearXNG       — self-hosted, no key, no limits, multi-engine aggregator
  2. Brave Search  — free tier 2000/month, independent index, clean JSON API
  3. Tavily        — agent-optimized, includes page content, free tier
  4. Exa           — neural/semantic search, free tier
  5. Groq compound — uses compound-beta model with built-in web search

Web fetch backends:
  1. Jina Reader   — r.jina.ai prefix, returns clean markdown, free, no key
  2. httpx fallback — raw HTML stripped (original approach)

Environment variables:
  SEARXNG_URL          — your SearXNG instance  e.g. http://localhost:8080
                         (if not set, SearXNG backend is skipped)
  BRAVE_SEARCH_KEY     — api.search.brave.com key (free tier: 2000 req/month)
                         https://api.search.brave.com/app/keys
  TAVILY_API_KEY       — tavily.com key (free tier: 1000 req/month)
                         https://app.tavily.com
  EXA_API_KEY          — exa.ai key (free tier available)
                         https://dashboard.exa.ai
  GROQ_API_KEY         — groq.com key (free tier available)
                         https://console.groq.com

Running SearXNG locally (recommended — Docker one-liner):
  docker run -d -p 8080:8080 \\
    -e SEARXNG_SECRET_KEY=$(openssl rand -hex 32) \\
    --name searxng searxng/searxng:latest

  Then set: export SEARXNG_URL=http://localhost:8080
"""

from __future__ import annotations

import os
import re
import json
from typing import Optional

import httpx
from groq import AsyncGroq

from tool_registry import Tool, ToolRegistry


# ─── Config ───────────────────────────────────────────────────────────────────

SEARXNG_URL  = os.getenv("SEARXNG_URL", "").rstrip("/")
BRAVE_KEY    = os.getenv("BRAVE_SEARCH_KEY", "")
TAVILY_KEY   = os.getenv("TAVILY_API_KEY", "")
EXA_KEY      = os.getenv("EXA_API_KEY", "")
GROQ_KEY     = os.getenv("GROQ_API_KEY", "")

HTTP_TIMEOUT = 20.0
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/120 Safari/537.36"
    )
}

# ── Groq compound-beta model name ─────────────────────────────────────────────
# BUG FIX: was "groq/compound" — that is not a valid model name and causes 413.
# The correct model strings are "compound-beta" and "compound-beta-mini".
GROQ_COMPOUND_MODEL = "compound-beta"


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_executed_tool_results(tool) -> list[dict]:
    """
    Parse search results from a Groq executed tool object.

    BUG FIX: tool.search_results is an ExecutedToolSearchResults *object*, not a
    plain dict.  Calling .get("results", []) on it raises:
        AttributeError: 'ExecutedToolSearchResults' object has no attribute 'get'

    We handle the object form (attribute access) with a fallback to dict form
    so this works regardless of future SDK changes.
    """
    sr = getattr(tool, "search_results", None)
    if sr is None:
        return []

    # Case 1: plain dict (future-proof / older SDK versions)
    if isinstance(sr, dict):
        raw = sr.get("results", [])
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("content", r.get("snippet", "")),
            }
            for r in raw
        ]

    # Case 2: ExecutedToolSearchResults object — access .results attribute
    raw = getattr(sr, "results", None) or []
    results = []
    for r in raw:
        if isinstance(r, dict):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("content", r.get("snippet", "")),
            })
        else:
            # SDK result object with attributes
            results.append({
                "title":   getattr(r, "title", ""),
                "url":     getattr(r, "url", ""),
                "snippet": getattr(r, "content", getattr(r, "snippet", "")),
            })
    return results


async def _search_groq(query: str, num_results: int) -> Optional[list[dict]]:
    """
    Groq compound-beta built-in web search.
    The compound-beta model has native web-search capability baked in.
    """
    if not GROQ_KEY:
        return None
    try:
        client = AsyncGroq(api_key=GROQ_KEY)

        # SANITIZATION: Force strict search mode via system prompt
        print(f"[INTERNAL DETECTION] Sanitizing Groq search input for: {query[:50]}...")
        messages = [
            {"role": "system", "content": "You are a robotic search utility. 1. Perform a search. 2. Output ONLY the tool results. 3. If tools fail, output a list of URLs and titles in JSON format: [{\"title\":\"...\",\"url\":\"...\",\"snippet\":\"...\"}]. Do not provide prose."},
            {"role": "user", "content": f"Search for: {query[:500]}"}
        ]

        response = await client.chat.completions.create(
            model=GROQ_COMPOUND_MODEL,
            messages=messages,
            temperature=0.0, # minimal creativity
        )

        msg = response.choices[0].message
        executed = getattr(msg, "executed_tools", None) or []

        results = []
        for tool in executed:
            parsed = _parse_executed_tool_results(tool)
            results.extend(parsed)
            # Remove the 'break' to capture results from multiple tools if present

        if results:
            print(f"[INTERNAL DETECTION] Groq structured extraction success. Extracted {len(results)} items.")
            return [
                {**r, "source": "groq"}
                for r in results[:num_results]
            ]
        
        # FAILSAFE: If no tools, try to parse JSON or URLs from msg.content
        content = getattr(msg, "content", "") or ""
        if not results and content:
            print(f"[INTERNAL DETECTION] No tool results. Attempting fallback extraction from prose ({len(content)} chars)...")
            # 1. Try to find a JSON block
            json_match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
            if json_match:
                try:
                    results = json.loads(json_match.group(0))
                    print("[INTERNAL DETECTION] Failsafe: Successfully parsed JSON from prose.")
                except:
                    pass
            
            # 2. Try to find Markdown links [title](url)
            if not results:
                links = re.findall(r"\[([^\]]+)\]\((https?://[^\)]+)\)", content)
                for title, url in links:
                    results.append({"title": title, "url": url, "snippet": ""})
                if results:
                    print(f"[INTERNAL DETECTION] Failsafe: Extracted {len(results)} links from prose.")

        if results:
            print(f"[INTERNAL DETECTION] Groq structured/failsafe extraction success. Items: {len(results)}")
            return [
                {**r, "source": "groq"}
                for r in results[:num_results]
            ]
        
        print(f"[INTERNAL DETECTION] Groq returned unusable content. Discarding for sanitation.")
        return None

    except Exception as e:
        msg = f"Groq Search failed: {e}"
        print(f"[web_search] {msg}")
        # Return error marker so caller knows Groq was tried but failed
        return [{"title": "Error", "url": "", "snippet": msg, "source": "groq-error"}]


async def _search_searxng(query: str, num_results: int) -> Optional[list[dict]]:
    """
    SearXNG JSON API.
    Aggregates Google, Bing, DuckDuckGo, Brave, Wikipedia, etc. simultaneously.
    Run locally — completely private, no external API key.
    """
    if not SEARXNG_URL:
        return None
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={
                    "q":         query,
                    "format":    "json",
                    "engines":   "google,bing,duckduckgo,brave,wikipedia",
                    "language":  "en",
                    "safesearch": "0",
                },
                headers=HTTP_HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for r in data.get("results", [])[:num_results]:
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("content", ""),
                "source":  "searxng",
            })
        return results if results else None

    except Exception as e:
        msg = f"SearXNG failed: {e}"
        print(f"[web_search] {msg}")
        return [{"title": "Error", "url": "", "snippet": msg, "source": "searxng-error"}]


async def _search_brave(query: str, num_results: int) -> Optional[list[dict]]:
    """
    Brave Search API.
    Independent index (not Bing-resold). Free tier: 2000 req/month.
    Get key: https://api.search.brave.com/app/keys
    """
    if not BRAVE_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={
                    "q":     query,
                    "count": min(num_results, 20),
                },
                headers={
                    "Accept":               "application/json",
                    "Accept-Encoding":      "gzip",
                    "X-Subscription-Token": BRAVE_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        if data.get("query", {}).get("answer"):
            results.append({
                "title":   "Featured Answer",
                "url":     "",
                "snippet": data["query"]["answer"],
                "source":  "brave-answer",
            })

        for r in data.get("web", {}).get("results", [])[:num_results]:
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("description", ""),
                "source":  "brave",
            })

        return results if results else None

    except Exception as e:
        msg = f"Brave failed: {e}"
        print(f"[web_search] {msg}")
        return [{"title": "Error", "url": "", "snippet": msg, "source": "brave-error"}]


async def _search_tavily(query: str, num_results: int) -> Optional[list[dict]]:
    """
    Tavily Search API.
    Purpose-built for AI agents — returns results with full page content.
    Free tier: 1000 req/month. https://app.tavily.com
    """
    if not TAVILY_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":              TAVILY_KEY,
                    "query":                query,
                    "search_depth":         "basic",
                    "max_results":          min(num_results, 10),
                    "include_answer":       True,
                    "include_raw_content":  False,
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        if data.get("answer"):
            results.append({
                "title":   "Tavily Answer",
                "url":     "",
                "snippet": data["answer"],
                "source":  "tavily-answer",
            })
        for r in data.get("results", [])[:num_results]:
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("content", ""),
                "score":   r.get("score", 0),
                "source":  "tavily",
            })

        return results if results else None

    except Exception as e:
        msg = f"Tavily failed: {e}"
        print(f"[web_search] {msg}")
        return [{"title": "Error", "url": "", "snippet": msg, "source": "tavily-error"}]


async def _search_exa(query: str, num_results: int) -> Optional[list[dict]]:
    """
    Exa (formerly Metaphor) — neural/semantic search.
    Free tier available. https://dashboard.exa.ai
    """
    if not EXA_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                "https://api.exa.ai/search",
                json={
                    "query":         query,
                    "numResults":    min(num_results, 10),
                    "useAutoprompt": True,
                    "contents": {
                        "text":    {"maxCharacters": 500},
                        "summary": {"query": query},
                    },
                },
                headers={
                    "x-api-key":    EXA_KEY,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for r in data.get("results", [])[:num_results]:
            snippet = (
                r.get("summary")
                or r.get("text", "")[:400]
                or (r.get("highlights") or [""])[0]
            )
            results.append({
                "title":     r.get("title", ""),
                "url":       r.get("url", ""),
                "snippet":   snippet,
                "published": r.get("publishedDate", ""),
                "source":    "exa",
            })
        return results if results else None

    except Exception as e:
        msg = f"Exa failed: {e}"
        print(f"[web_search] {msg}")
        return [{"title": "Error", "url": "", "snippet": msg, "source": "exa-error"}]


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED SEARCH TOOL  —  auto-fallback chain
# ══════════════════════════════════════════════════════════════════════════════

async def _web_search(query: str, num_results: int = 8, backend: str = "auto") -> str:
    """
    Web search with automatic fallback chain.
    backend: "auto" | "searxng" | "brave" | "tavily" | "exa" | "groq"

    Auto priority: SearXNG → Brave → Tavily → Exa → Groq
    (SearXNG first because it's local and unlimited; Groq last to save tokens)
    """
    print(f"[INTERNAL DETECTION] Starting search for: '{query}' (backend: {backend})")
    num_results = int(num_results)
    results: Optional[list[dict]] = None
    used_backend = "none"

    if backend == "auto":
        backends = [
            ("searxng", _search_searxng),
            ("brave",   _search_brave),
            ("tavily",  _search_tavily),
            ("exa",     _search_exa),
            ("groq",    _search_groq),
        ]
        for name, fn in backends:
            print(f"[INTERNAL DETECTION] Trying search backend: {name}")
            candidate = await fn(query, num_results)
            # Skip error-only results and try next backend
            if candidate and not all(r.get("source", "").endswith("-error") for r in candidate):
                results = candidate
                used_backend = name
                print(f"[INTERNAL DETECTION] Search success via {name}. Found {len(results)} results.")
                break
            if candidate:
                print(f"[INTERNAL DETECTION] Backend {name} returned error or no results.")
                # All were errors — keep them to report but keep trying
                results = results or candidate
                used_backend = name
    else:
        fn_map = {
            "groq":    _search_groq,
            "searxng": _search_searxng,
            "brave":   _search_brave,
            "tavily":  _search_tavily,
            "exa":     _search_exa,
        }
        if backend not in fn_map:
            return (
                f"web_search: unknown backend '{backend}'. "
                "Valid: auto, groq, searxng, brave, tavily, exa"
            )
        results = await fn_map[backend](query, num_results)
        used_backend = backend

    if not results:
        configured = []
        if SEARXNG_URL: configured.append("searxng")
        if BRAVE_KEY:   configured.append("brave")
        if TAVILY_KEY:  configured.append("tavily")
        if EXA_KEY:     configured.append("exa")
        if GROQ_KEY:    configured.append("groq")

        if not configured:
            return (
                "web_search: no backends configured.\n"
                "Set one of: SEARXNG_URL, BRAVE_SEARCH_KEY, TAVILY_API_KEY, "
                "EXA_API_KEY, GROQ_API_KEY\n"
                "Quickest start (local, no key): "
                "docker run -d -p 8080:8080 searxng/searxng && "
                "export SEARXNG_URL=http://localhost:8080"
            )
        return (
            f"web_search: all configured backends ({', '.join(configured)}) "
            f"returned no results for: {query}"
        )

    # Format output
    lines = [f"Search: {query}  [via {used_backend}]\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        if r.get("url"):
            lines.append(f"    {r['url']}")
        if r.get("snippet"):
            lines.append(f"    {r['snippet'][:300]}")
        if r.get("published"):
            lines.append(f"    Published: {r['published'][:10]}")
        lines.append("")

    return "\n".join(lines)


WEB_SEARCH_TOOL = Tool(
    name="web_search",
    description=(
        "Search the web for current information. "
        "Tries SearXNG (self-hosted) → Brave → Tavily → Exa → Groq in order. "
        "Returns titles, URLs, and content snippets. "
        "Use for recent news, current facts, or anything beyond training data."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific for better results.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default 8, max 20)",
                "default": 8,
            },
            "backend": {
                "type": "string",
                "description": "Force a specific backend: auto (default), searxng, brave, tavily, exa, groq",
                "default": "auto",
            },
        },
        "required": ["query"],
    },
    handler=_web_search,
    tags=["web", "search"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  WEB FETCH  —  Jina Reader + httpx fallback
# ══════════════════════════════════════════════════════════════════════════════

async def _fetch_groq(url: str, max_chars: int) -> Optional[str]:
    """
    Groq compound-beta built-in URL visit.
    BUG FIX: was "groq/compound" → correct name is "compound-beta"
    """
    if not GROQ_KEY:
        return None
    try:
        client = AsyncGroq(api_key=GROQ_KEY)
        response = await client.chat.completions.create(
            model=GROQ_COMPOUND_MODEL,
            messages=[{"role": "user", "content": f"Visit and extract the main content of: {url}"}],
        )

        msg = response.choices[0].message
        executed = getattr(msg, "executed_tools", None) or []

        for tool in executed:
            tool_type = getattr(tool, "type", "")
            if "visit" in tool_type or "fetch" in tool_type or "browse" in tool_type:
                content = getattr(tool, "output", "") or getattr(tool, "content", "")
                if content:
                    return str(content)[:max_chars]

        # If no visit tool found, the model's text response may contain the content
        text = getattr(msg, "content", "") or ""
        if text and len(text) > 200:
            return text[:max_chars]

        return None
    except Exception as e:
        print(f"[web_fetch] Groq Visit failed: {e}")
        return None


async def _web_fetch(url: str, max_chars: int = 8000, use_jina: bool = True) -> str:
    """
    Fetch clean readable content from a URL.

    Primary:  Jina Reader (r.jina.ai prefix) — free, no key, clean markdown
    Secondary: Groq compound-beta URL visit (if key configured)
    Fallback: httpx + regex HTML strip
    """
    print(f"[INTERNAL DETECTION] Starting fetch for URL: {url} (use_jina: {use_jina})")
    max_chars = int(max_chars)
    if isinstance(use_jina, str):
        use_jina = use_jina.lower() != "false"

    if not url.startswith(("http://", "https://")):
        return f"web_fetch: invalid URL '{url}' — must start with http:// or https://"

    # ── Jina Reader (Primary — free, best quality) ───────────────────────────
    if use_jina:
        jina_url = f"https://r.jina.ai/{url}"
        try:
            async with httpx.AsyncClient(
                timeout=HTTP_TIMEOUT,
                follow_redirects=True,
                headers={
                    **HTTP_HEADERS,
                    "X-Return-Format": "markdown",
                    "X-Timeout":       "15",
                },
            ) as client:
                resp = await client.get(jina_url)
                resp.raise_for_status()
                text = resp.text.strip()

            if text and len(text) > 100:
                suffix = (
                    f"\n\n[truncated — {len(text) - max_chars} more chars]"
                    if len(text) > max_chars else ""
                )
                return text[:max_chars] + suffix
        except Exception as e:
            print(f"[web_fetch] Jina Reader failed ({e}), falling back")

    # ── Groq Visit (Secondary) ────────────────────────────────────────────────
    if GROQ_KEY:
        content = await _fetch_groq(url, max_chars)
        if content:
            return content

    # ── httpx + HTML strip (Fallback) ─────────────────────────────────────────
    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            headers=HTTP_HEADERS,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "json" in content_type:
                return json.dumps(resp.json(), indent=2)[:max_chars]

            html = resp.text

        html = re.sub(
            r"<(script|style|noscript|nav|footer|header|aside)[^>]*>.*?</\1>",
            "", html, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        suffix = (
            f"\n\n[truncated — {len(text) - max_chars} more chars]"
            if len(text) > max_chars else ""
        )
        return text[:max_chars] + suffix

    except httpx.HTTPStatusError as e:
        return f"web_fetch: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"web_fetch error: {e}"


WEB_FETCH_TOOL = Tool(
    name="web_fetch",
    description=(
        "Fetch the full readable content of a URL as clean text or markdown. "
        "Primary: Jina Reader (r.jina.ai) — free, no key, returns clean markdown. "
        "Secondary: Groq compound-beta URL visit (if key set). "
        "Fallback: direct HTTP + HTML stripping. "
        "Use when you need to read an entire page referenced in search results."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL to fetch (must start with http:// or https://)",
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return (default 8000)",
                "default": 8000,
            },
            "use_jina": {
                "type": "boolean",
                "description": "Use Jina Reader for clean markdown (default true)",
                "default": True,
            },
        },
        "required": ["url"],
    },
    handler=_web_fetch,
    tags=["web", "fetch"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  Registration helper
# ══════════════════════════════════════════════════════════════════════════════

def register_web_tools(registry: ToolRegistry) -> None:
    """
    Register the production web search and fetch tools.
    Call AFTER register_all_tools() to override the default DDG implementations.
    """
    registry.register(WEB_SEARCH_TOOL)
    registry.register(WEB_FETCH_TOOL)
    print("[web_tools] Registered: web_search, web_fetch")
    _print_backend_status()


def _print_backend_status():
    """Print which backends are configured on startup."""
    print("[web_tools] Backend status:")
    print(f"  Groq     : {'✓  key configured (compound-beta)' if GROQ_KEY else '✗  not configured (set GROQ_API_KEY)'}")
    print(f"  SearXNG  : {'✓ ' + SEARXNG_URL if SEARXNG_URL else '✗  not configured (set SEARXNG_URL)'}")
    print(f"  Brave    : {'✓  key configured' if BRAVE_KEY else '✗  not configured (set BRAVE_SEARCH_KEY)'}")
    print(f"  Tavily   : {'✓  key configured' if TAVILY_KEY else '✗  not configured (set TAVILY_API_KEY)'}")
    print(f"  Exa      : {'✓  key configured' if EXA_KEY else '✗  not configured (set EXA_API_KEY)'}")
    print(f"  Jina     : ✓  always available (no key required)")