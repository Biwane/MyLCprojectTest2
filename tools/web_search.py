"""
Web Search Tool Module

This module provides tools for searching the web, retrieving, and processing web content.
It integrates with search providers and offers web scraping capabilities to fetch
and extract information from websites.
"""

import os
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union
import urllib.parse
import re
import html

import requests
from bs4 import BeautifulSoup

# Optional import for Tavily API if available
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

logger = logging.getLogger(__name__)

class WebSearchTool:
    """
    Tool for searching the web and retrieving relevant information.
    
    This tool integrates with search providers like Tavily or falls back to
    a basic web search implementation when specialized APIs aren't available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the web search tool.
        
        Args:
            config: Configuration dictionary with search tool settings
        """
        self.config = config
        self.search_provider = config.get("search_provider", "tavily")
        self.max_results = config.get("max_results", 5)
        self.search_timeout = config.get("search_timeout", 30)
        self.enable_scraping = config.get("enable_scraping", True)
        self.user_agent = config.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Initialize search providers
        self._initialize_search_providers()
        
        logger.debug(f"Initialized WebSearchTool with provider: {self.search_provider}")
    
    def _initialize_search_providers(self):
        """Initialize the configured search providers."""
        # Initialize Tavily if available and configured
        self.tavily_client = None
        if self.search_provider == "tavily":
            if TAVILY_AVAILABLE:
                api_key = self.config.get("tavily_api_key") or os.getenv("TAVILY_API_KEY")
                if api_key:
                    try:
                        self.tavily_client = TavilyClient(api_key=api_key)
                        logger.info("Initialized Tavily search client")
                    except Exception as e:
                        logger.error(f"Error initializing Tavily client: {str(e)}")
                        self.search_provider = "basic"
                else:
                    logger.warning("Tavily API key not found, falling back to basic search")
                    self.search_provider = "basic"
            else:
                logger.warning("Tavily package not available, falling back to basic search")
                self.search_provider = "basic"
    
    def search(self, query: str, max_results: int = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Search the web for the given query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return (overrides config)
            **kwargs: Additional search parameters
            
        Returns:
            List of search results with URL, title, and snippet
        """
        max_results = max_results or self.max_results
        
        # Log the search
        logger.info(f"Searching for: {query} (provider: {self.search_provider})")
        
        try:
            # Use the appropriate search provider
            if self.search_provider == "tavily" and self.tavily_client:
                return self._search_tavily(query, max_results, **kwargs)
            else:
                return self._search_basic(query, max_results, **kwargs)
                
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return [{
                "url": "",
                "title": "Error during search",
                "content": f"An error occurred: {str(e)}",
                "source": "error"
            }]
    
    def _search_tavily(self, query: str, max_results: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Search using the Tavily API.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            **kwargs: Additional Tavily-specific parameters
            
        Returns:
            List of search results
        """
        include_answer = kwargs.get("include_answer", True)
        search_depth = kwargs.get("search_depth", "basic")
        
        try:
            # Execute the search
            search_result = self.tavily_client.search(
                query=query, 
                search_depth=search_depth,
                max_results=max_results,
                include_answer=include_answer
            )
            
            # Extract the results
            results = []
            
            # Add the Tavily-generated answer if available
            if include_answer and "answer" in search_result and search_result["answer"]:
                results.append({
                    "url": "",
                    "title": "AI-Generated Answer",
                    "content": search_result["answer"],
                    "source": "tavily_answer"
                })
            
            # Add the individual search results
            for result in search_result.get("results", []):
                results.append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "source": "tavily"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error with Tavily search: {str(e)}")
            # Fall back to basic search
            logger.info("Falling back to basic search")
            return self._search_basic(query, max_results, **kwargs)
    
    def _search_basic(self, query: str, max_results: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Basic web search implementation using a public search API or direct requests.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            **kwargs: Additional parameters
            
        Returns:
            List of search results
        """
        # This is a placeholder for a basic search implementation
        # In a production environment, you would integrate with a public search API
        
        # Encode the query for URL
        encoded_query = urllib.parse.quote(query)
        
        # We'll use a publicly accessible search service for demonstration
        # Note: This is not a reliable or production-ready approach
        search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        try:
            # Send the request
            headers = {"User-Agent": self.user_agent}
            response = requests.get(search_url, headers=headers, timeout=self.search_timeout)
            response.raise_for_status()
            
            # Parse the response
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract search results
            results = []
            result_elements = soup.select(".result")[:max_results]
            
            for element in result_elements:
                title_elem = element.select_one(".result__title")
                link_elem = element.select_one(".result__url")
                snippet_elem = element.select_one(".result__snippet")
                
                title = title_elem.get_text().strip() if title_elem else "No title"
                url = link_elem.get_text().strip() if link_elem else ""
                snippet = snippet_elem.get_text().strip() if snippet_elem else "No snippet available"
                
                # Clean up the URL
                if url and not url.startswith(("http://", "https://")):
                    url = "https://" + url
                
                results.append({
                    "url": url,
                    "title": title,
                    "content": snippet,
                    "source": "basic_search"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error with basic search: {str(e)}")
            # Return an empty result with the error
            return [{
                "url": "",
                "title": "Search failed",
                "content": f"The search failed with error: {str(e)}",
                "source": "error"
            }]
    
    def get_webpage_content(self, url: str, extract_main_content: bool = True) -> Dict[str, Any]:
        """
        Retrieve and extract content from a webpage.
        
        Args:
            url: The URL of the webpage to retrieve
            extract_main_content: Whether to extract just the main content (vs. entire HTML)
            
        Returns:
            Dictionary with URL, title, and content
        """
        if not url or not url.startswith(("http://", "https://")):
            return {
                "url": url,
                "title": "Invalid URL",
                "content": "The provided URL is invalid or empty.",
                "success": False
            }
        
        try:
            # Send the request
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers, timeout=self.search_timeout)
            response.raise_for_status()
            
            # Parse the response
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            title = soup.title.get_text() if soup.title else "No title"
            
            if extract_main_content:
                # Extract the main content
                # This is a simplified approach and may not work for all websites
                # A production implementation would use more sophisticated content extraction
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    script.extract()
                
                # Find the main content
                main_content = None
                
                # Try common content containers
                for container in ["main", "article", "div[role='main']", "#content", ".content", "#main", ".main"]:
                    content_elem = soup.select_one(container)
                    if content_elem and len(content_elem.get_text(strip=True)) > 200:
                        main_content = content_elem
                        break
                
                # If no main content found, use the body
                if not main_content:
                    main_content = soup.body
                
                # Extract text content
                if main_content:
                    paragraphs = main_content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])
                    content = "\n".join([p.get_text().strip() for p in paragraphs])
                else:
                    # Fallback to raw text from body
                    content = soup.body.get_text(separator="\n", strip=True)
            else:
                # Use the entire HTML
                content = str(soup)
            
            # Clean up the content
            content = re.sub(r'\s+', ' ', content).strip()
            content = html.unescape(content)
            
            return {
                "url": url,
                "title": title,
                "content": content,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error retrieving webpage content from {url}: {str(e)}")
            return {
                "url": url,
                "title": "Error retrieving content",
                "content": f"An error occurred: {str(e)}",
                "success": False
            }
    
    def search_and_summarize(
        self, 
        query: str, 
        max_results: int = None,
        summarize_results: bool = True,
        fetch_full_content: bool = False
    ) -> Dict[str, Any]:
        """
        Search the web and optionally summarize the results.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            summarize_results: Whether to summarize the results
            fetch_full_content: Whether to fetch full content for each result
            
        Returns:
            Dictionary with search results and optionally a summary
        """
        # Perform the search
        search_results = self.search(query, max_results)
        
        # Fetch full content if requested
        if fetch_full_content and self.enable_scraping:
            for i, result in enumerate(search_results):
                if result.get("url") and result["source"] != "error" and result["source"] != "tavily_answer":
                    page_content = self.get_webpage_content(result["url"])
                    if page_content["success"]:
                        search_results[i]["content"] = page_content["content"]
        
        response = {
            "query": query,
            "results": search_results,
            "timestamp": time.time()
        }
        
        # No built-in summarization in this simplified version
        if summarize_results:
            response["summary"] = "Summarization capability requires integration with an LLM."
        
        return response
