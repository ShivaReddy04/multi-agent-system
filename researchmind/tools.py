from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information.
    Returns Titles, URLs and snippets.
    """

    results = tavily.search(
        query=query,
        max_results=5
    )

    out = []

    for r in results["results"]:
        title = r.get("title", "No Title")
        url = r.get("url", "")
        snippet = r.get("content", "")[:200]

        out.append(
            f"""
Title: {title}
URL: {url}
Snippet: {snippet}
"""
        )

    return "\n--------------------\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """
    Scrape and return text content from a URL.
    """

    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        soup = BeautifulSoup(
            resp.text,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header"
        ]):
            tag.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True
        )

        return text[:1000]

    except Exception as e:
        return f"Could not scrape {url}. Error: {str(e)}"
