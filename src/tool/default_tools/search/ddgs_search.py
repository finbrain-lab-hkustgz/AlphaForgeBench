from typing import List, Optional, Dict, Any
import json
import asyncio

from ddgs import DDGS
from pydantic import Field

from src.tool.default_tools.search.types import SearchItem
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.logger import logger
from src.registry import TOOL

@TOOL.register_module(force=True)
class DDGSSearch(Tool):
    """A tool for searching using DDGS search engine."""
    
    name: str = "ddgs_search"
    description: str = (
        "a search engine using DDGS. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")

    async def _perform_search(
        self, 
        query: str, 
        num_results: int = 10,
        region: Optional[str] = "wt-wt",
        safesearch: Optional[str] = "moderate",
        *args, 
        **kwargs
    ) -> List[SearchItem]:
        """
        DDGS search engine using ddgs library.

        Returns results formatted according to SearchItem model.
        """
        # Run DDGS in thread pool to avoid blocking (DDGS is synchronous)
        def _search_sync():
            results = []
            with DDGS() as ddgs:
                raw_results = ddgs.text(
                    query,
                    max_results=num_results,
                    region=region or "wt-wt",
                    safesearch=safesearch or "moderate",
                )
                for item in raw_results:
                    if isinstance(item, dict):
                        results.append(
                            SearchItem(
                                title=item.get("title", f"DuckDuckGo Result {len(results) + 1}"),
                                url=item.get("href", ""),
                                description=item.get("body", None),
                            )
                        )
                    elif isinstance(item, str):
                        # If it's just a URL
                        results.append(
                            SearchItem(
                                title=f"DuckDuckGo Result {len(results) + 1}",
                                url=item,
                                description=None
                            )
                        )
            return results
        
        # Run sync search in thread pool to avoid blocking event loop
        return await asyncio.to_thread(_search_sync)

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
        DuckDuckGo search tool.
        
        Args:
            query (str): The query to search for.
            num_results (Optional[int]): The number of search results to return.
            country (Optional[str]): The country to search in (maps to region parameter, e.g., "us" -> "us-en", "sg" -> "sg-en").
            lang (Optional[str]): The language to search in (maps to region parameter, e.g., "en" -> "us-en", "zh" -> "cn-zh").
            filter_year (Optional[int]): The year to filter results by (not used by DuckDuckGo, kept for compatibility).
        """
        try:
            # Map country/lang to region format
            # Default region mapping: country-lang
            region = "wt-wt"  # Default to worldwide
            if country and lang:
                # Try to map to region format (e.g., "us-en", "sg-en", "cn-zh")
                region_map = {
                    ("us", "en"): "us-en",
                    ("sg", "en"): "sg-en",
                    ("cn", "zh"): "cn-zh",
                    ("uk", "en"): "uk-en",
                }
                region = region_map.get((country.lower(), lang.lower()), f"{country.lower()}-{lang.lower()}")
            elif country:
                region = f"{country.lower()}-en"
            elif lang:
                region = f"wt-{lang.lower()}"
            
            # Perform search
            search_items = await self._perform_search(
                query, 
                num_results=num_results,
                region=region,
                safesearch="moderate"
            )
            
            # Format results as JSON string
            results_json = json.dumps([{
                "title": item.title,
                "url": item.url,
                "description": item.description or ""
            } for item in search_items], ensure_ascii=False, indent=4)
            
            message = f"DuckDuckGo search results for query: {query}\n\n{results_json}"
            
            return ToolResponse(
                success=True, 
                message=message, 
                extra=ToolExtra(
                    data={
                        "query": query,
                        "num_results": len(search_items),
                        "search_items": search_items,
                        "engine": "duckduckgo"
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {e}")
            return ToolResponse(
                success=False, 
                message=f"Error in DuckDuckGo search: {str(e)}",
            )