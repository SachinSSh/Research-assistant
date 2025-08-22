import asyncio
import aiohttp
import time
from typing import List, Optional, Dict, Any
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from langchain_community.tools import TavilySearchResults
from ..config import settings
from ..models.schemas import SearchResult
from ..utils.monitoring import track_api_call


class SearchService:
    def __init__(self):
        self.tavily_search = None
        if settings.tavily_api_key:
            self.tavily_search = TavilySearchResults(
                api_key=settings.tavily_api_key,
                max_results=settings.max_search_results
            )
        
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.request_timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @track_api_call
    async def search_web(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """Perform web search using available search providers."""
        max_results = max_results or settings.max_search_results
        
        if self.tavily_search:
            return await self._search_with_tavily(query, max_results)
        elif settings.serp_api_key:
            return await self._search_with_serp(query, max_results)
        else:
            return await self._fallback_search(query, max_results)

    async def _search_with_tavily(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Tavily API."""
        try:
            results = await asyncio.to_thread(
                self.tavily_search.invoke, 
                {"query": query}
            )
            
            search_results = []
            for i, result in enumerate(results[:max_results]):
                search_results.append(SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("content", ""),
                    relevance_score=1.0 - (i * 0.1)  
                ))
            
            return search_results
        except Exception as e:
            raise Exception(f"Tavily search failed: {str(e)}")

    async def _search_with_serp(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using SERP API."""
        url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "api_key": settings.serp_api_key,
            "engine": "google",
            "num": max_results
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            
            search_results = []
            for i, result in enumerate(data.get("organic_results", [])[:max_results]):
                search_results.append(SearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    relevance_score=1.0 - (i * 0.1)
                ))
            
            return search_results

    async def _fallback_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback search using DuckDuckGo (simplified)."""
        encoded_query = quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={encoded_query}"
        
        async with self.session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            results = []
            result_elements = soup.find_all('div', class_='result')[:max_results]
            
            for i, element in enumerate(result_elements):
                title_elem = element.find('a', class_='result__a')
                snippet_elem = element.find('div', class_='result__snippet')
                
                if title_elem and snippet_elem:
                    results.append(SearchResult(
                        title=title_elem.get_text(strip=True),
                        url=title_elem.get('href', ''),
                        snippet=snippet_elem.get_text(strip=True),
                        relevance_score=1.0 - (i * 0.1)
                    ))
            
            return results

    async def fetch_content(self, url: str) -> str:
        """Fetch and extract text content from a URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    text = soup.get_text()
                    
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    return text[:settings.max_content_length]
                else:
                    return f"Failed to fetch content: HTTP {response.status}"
                    
        except Exception as e:
            return f"Error fetching content: {str(e)}"

    async def batch_fetch_content(self, urls: List[str], max_concurrency: int = 5) -> Dict[str, str]:
        """Fetch content from multiple URLs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def fetch_single(url: str) -> tuple[str, str]:
            async with semaphore:
                content = await self.fetch_content(url)
                return url, content
        
        tasks = [fetch_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        content_dict = {}
        for result in results:
            if isinstance(result, tuple):
                url, content = result
                content_dict[url] = content
            else:
                continue
        
        return content_dict