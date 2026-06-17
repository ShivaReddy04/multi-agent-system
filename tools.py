from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import  os
from dotenv import load_dotenv
from rich import print
load_dotenv()


tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information.Returns Titles, URLs and snippets."""
    results = tavily.search(query=query,max_results=5)
   
    out = []
    for r in results['results']:
        out.append(
            f"Title : {r['title']}\nURL: {r['url']}\nSnippet: {r['content'] [:300]}\n"
        )
    return "\n----\n".join(out)






@tool
def scrape_url(url: str) -> str:
    """Scrape and return the text content from a given url for deeper understanding."""
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:1000]
    except Exception as e:
        return f"could not scrape {url} due to {str(e)}"

