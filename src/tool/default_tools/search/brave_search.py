from __future__ import annotations
from typing import Any, Optional, Dict, List
import json
import os
from pydantic import ConfigDict, Field
import aiohttp
from dotenv import load_dotenv
load_dotenv()

from src.tool.default_tools.search.types import SearchItem, SearchToolArgs
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.logger import logger
from src.registry import TOOL

@TOOL.register_module(force=True)
class BraveSearch(Tool):
    """Tool that queries the Brave search engine.

    Example usages:
    .. code-block:: python
        # basic usage
        tool = BraveSearch()

    .. code-block:: python
        # with custom search kwargs
        tool = BraveSearch.from_search_kwargs({"count": 5})
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "brave_search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    api_key: Optional[str] = Field(default=None, description="Brave Search API key")
    base_url: str = "https://api.search.brave.com/res/v1/web/search"
    
    def __init__(self, **kwargs):
        """Initialize the BraveSearch tool."""
        # Set api_key from environment if not provided
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("BRAVE_SEARCH_API_KEY")

    @classmethod
    def from_search_kwargs(cls, search_kwargs: dict, **kwargs: Any) -> BraveSearch:
        """Create a tool from search kwargs.

        Args:
            search_kwargs: Any additional kwargs to pass to the search function.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A tool.
        """
        return cls(search_kwargs=search_kwargs, **kwargs)

    async def _search_brave(self, 
                            query: str, 
                            num_results: int = 10, 
                            country: str = "us",
                            lang: str = "en",
                            filter_year: Optional[int] = 2025) -> List[SearchItem]:
        """
        Perform a Brave search using the provided parameters.
        Returns a list of SearchItem objects.
        """
        if not self.api_key:
            raise ValueError("BRAVE_SEARCH_API_KEY environment variable is required")
        
        results = []
        
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        
        params = {
            "q": query,
            "count": num_results,
            "country": country.upper(),
            "search_lang": lang.lower(),
            "extra_snippets": "true",  # Convert bool to string for API
        }
        
        # Add date filter if year is valid (1900-2100)
        # Handle None case (when explicitly passed from caller)
        if filter_year is None:
            filter_year = 2025  # Use default if None
        
        if 1900 <= filter_year <= 2100:
            # Brave API uses different date filter format
            # Using safesearch parameter or custom date range if supported
            params["safesearch"] = "moderate"
            # Note: Brave API may not support direct year filtering like Firecrawl
            # This is a placeholder - adjust based on actual Brave API capabilities
        else:
            logger.warning(f"Invalid filter_year: {filter_year}. Expected 1900-2100. Ignoring date filter.")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, headers=headers, params=params) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise Exception(f"HTTP error {response.status}: {error_text}")
                    
                    data = await response.json()
                    web_results = data.get("web", {}).get("results", [])
                    
                    for item in web_results:
                        if item is None:
                            continue
                        
                        title = item.get("title", "") or ""
                        url = item.get("url", "") or ""
                        description_parts = [
                            item.get("description", ""),
                            *item.get("extra_snippets", [])
                        ]
                        description = " ".join(filter(None, description_parts)) or ""
                        
                        if url:  # Only add items with valid URLs
                            results.append(SearchItem(
                                title=title,
                                url=url,
                                description=description
                            ))
        except Exception as e:
            logger.error(f"Brave API call failed: {e}")
            return results

        return results
    
    async def __call__(
        self,
        query: str,
        num_results: Optional[int] = 5,
        country: Optional[str] = "us",
        lang: Optional[str] = "en",
        filter_year: Optional[int] = 2025,
        **kwargs
    ) -> ToolResponse:
        """
        Brave search tool.
        
        Args:
            query (str): The query to search for.
            num_results (Optional[int]): The number of search results to return.
            country (Optional[str]): The country to search in.
            lang (Optional[str]): The language to search in.
            filter_year (int): The year to filter results by. Defaults to 2025.
        """
        
        try:
            # Perform search
            search_items = await self._search_brave(
                query, 
                num_results=num_results, 
                country=country,
                lang=lang,
                filter_year=filter_year
            )
            
            # Format results as JSON string
            results_json = json.dumps([{
                "title": item.title,
                "url": item.url,
                "description": item.description or ""
            } for item in search_items], ensure_ascii=False, indent=4)
            
            message = f"Brave search results for query: {query}\n\n{results_json}"
            
            return ToolResponse(success=True, message=message, extra=ToolExtra(
                data={
                    "query": query,
                    "num_results": len(search_items),
                    "search_items": search_items,
                    "engine": "brave"
                }
            ))
            
        except Exception as e:
            logger.error(f"Error in Brave search: {e}")
            return ToolResponse(
                success=False, 
                message=f"Error in Brave search: {str(e)}",
            )
