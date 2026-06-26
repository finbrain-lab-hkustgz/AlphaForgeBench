from typing import List, Optional, Tuple, Dict, Any
import json
import asyncio

import aiohttp
from bs4 import BeautifulSoup
from pydantic import Field

from src.tool.default_tools.search.types import SearchItem
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.logger import logger
from src.registry import TOOL


ABSTRACT_MAX_LENGTH = 300

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/49.0.2623.108 Chrome/49.0.2623.108 Safari/537.36",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; pt-BR) AppleWebKit/533.3 (KHTML, like Gecko) QtWeb Internet Browser/3.7 http://www.QtWeb.net",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.2 (KHTML, like Gecko) ChromePlus/4.0.222.3 Chrome/4.0.222.3 Safari/532.2",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.4pre) Gecko/20070404 K-Ninja/2.1.3",
    "Mozilla/5.0 (Future Star Technologies Corp.; Star-Blade OS; x86_64; U; en-US) iNet Browser 4.7",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.13) Gecko/20080414 Firefox/2.0.0.13 Pogo/2.0.0.13.6866",
]

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": USER_AGENTS[0],
    "Referer": "https://www.bing.com/",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

BING_HOST_URL = "https://www.bing.com"
BING_SEARCH_URL = "https://www.bing.com/search?q="


@TOOL.register_module(force=True)
class BingSearch(Tool):
    """A tool for searching using Bing search engine."""
    
    name: str = "bing_search"
    description: str = (
        "a search engine using Bing. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    session: Optional[aiohttp.ClientSession] = None

    def __init__(self, **kwargs):
        """Initialize the BingSearch tool."""
        super().__init__(**kwargs)

    async def _search(self, query: str, num_results: int = 10) -> List[SearchItem]:
        """
        Asynchronous Bing search implementation to retrieve search results.

        Args:
            query (str): The search query to submit to Bing.
            num_results (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            List[SearchItem]: A list of search items with title, URL, and description.
        """
        if not query:
            return []

        list_result = []
        first = 1
        next_url = BING_SEARCH_URL + query

        while len(list_result) < num_results:
            data, next_url = await self._parse_html(
                next_url, rank_start=len(list_result), first=first
            )
            if data:
                list_result.extend(data)
            if not next_url:
                break
            first += 10

        return list_result[:num_results]

    async def _parse_html(
        self, url: str, rank_start: int = 0, first: int = 1
    ) -> Tuple[List[SearchItem], str]:
        """
        Parse Bing search result HTML to extract search results and the next page URL.

        Returns:
            tuple: (List of SearchItem objects, next page URL or None)
        """
        try:
            # Create session if it doesn't exist
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession(headers=HEADERS)
            
            async with self.session.get(url=url) as res:
                res.raise_for_status()
                html_text = await res.text()
                root = BeautifulSoup(html_text, "lxml")

            list_data = []
            ol_results = root.find("ol", id="b_results")
            if not ol_results:
                return [], None

            for li in ol_results.find_all("li", class_="b_algo"):
                title = ""
                url = ""
                abstract = ""
                try:
                    h2 = li.find("h2")
                    if h2:
                        title = h2.text.strip()
                        url = h2.a["href"].strip()

                    p = li.find("p")
                    if p:
                        abstract = p.text.strip()

                    if ABSTRACT_MAX_LENGTH and len(abstract) > ABSTRACT_MAX_LENGTH:
                        abstract = abstract[:ABSTRACT_MAX_LENGTH]

                    rank_start += 1

                    # Create a SearchItem object
                    list_data.append(
                        SearchItem(
                            title=title or f"Bing Result {rank_start}",
                            url=url,
                            description=abstract,
                        )
                    )
                except Exception:
                    continue

            next_btn = root.find("a", title="Next page")
            if not next_btn:
                return list_data, None

            next_url = BING_HOST_URL + next_btn["href"]
            return list_data, next_url
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return [], None

    async def __call__(
        self,
        query: str,
        num_results: Optional[int] = 10,
        country: Optional[str] = "us",
        lang: Optional[str] = "en",
        filter_year: Optional[int] = None,
        **kwargs
    ) -> ToolResponse:
        """
        Bing search tool.
        
        Args:
            query (str): The query to search for.
            num_results (Optional[int]): The number of search results to return.
            country (Optional[str]): The country to search in (not used by Bing, kept for compatibility).
            lang (Optional[str]): The language to search in (not used by Bing, kept for compatibility).
            filter_year (Optional[int]): The year to filter results by (not used by Bing, kept for compatibility).
        """
        try:
            # Perform search asynchronously
            search_items = await self._search(query, num_results=num_results)
            
            # Format results as JSON string
            results_json = json.dumps([{
                "title": item.title,
                "url": item.url,
                "description": item.description or ""
            } for item in search_items], ensure_ascii=False, indent=4)
            
            message = f"Bing search results for query: {query}\n\n{results_json}"
            
            return ToolResponse(
                success=True, 
                message=message, 
                extra=ToolExtra(
                    data={
                        "query": query,
                        "num_results": len(search_items),
                        "search_items": search_items,
                        "engine": "bing"
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error in Bing search: {e}")
            return ToolResponse(
                success=False, 
                message=f"Error in Bing search: {str(e)}"
            )
