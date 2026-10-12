import json
import os
import urllib.error
import urllib.request


def search_google(query: str) -> dict:
    """Retrieve Google search snippets through Serper.

    Args:
        query: A short web search query.

    Returns:
        Search result titles, URLs and snippets, or an error.
    """
    key = os.environ.get("SERPER_API_KEY")

    if not key:
        return {"error": "SERPER_API_KEY is not configured."}

    if not query.strip() or len(query) > 500:
        return {"error": "Use a query of 1–500 characters."}

    request = urllib.request.Request(
        "https://google.serper.dev/search",
        data=json.dumps({
            "q": query,
            "num": 5,
        }).encode("utf-8"),
        headers={
            "X-API-KEY": key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:
            data = json.load(response)

    except urllib.error.HTTPError as exc:
        return {"error": f"Search service returned HTTP {exc.code}."}

    except (urllib.error.URLError, TimeoutError):
        return {"error": "Search service is unavailable."}

    except json.JSONDecodeError:
        return {"error": "Search service returned invalid JSON."}

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in data.get("organic", [])[:5]
    ]

    return {"query": query, "results": results}