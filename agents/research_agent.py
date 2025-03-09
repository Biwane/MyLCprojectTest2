"""
Research Agent Module

This module implements the ResearchAgent class, which specializes in gathering
and synthesizing information from various sources to support other agents.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union

from langchain_core.tools import BaseTool

from agents.base_agent import BaseAgent
from core.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """
    Agent specialized in gathering, analyzing, and synthesizing information.
    
    This agent can use web search tools, documentation retrieval, and other
    information-gathering methods to support the team's knowledge needs.
    """
    
    def __init__(
        self, 
        agent_executor,
        role: str = "research",
        config: Dict[str, Any] = None,
        knowledge_repository: Optional[KnowledgeRepository] = None
    ):
        """
        Initialize the research agent.
        
        Args:
            agent_executor: The LangChain agent executor
            role: The specific role of this research agent
            config: Configuration dictionary with agent settings
            knowledge_repository: Knowledge repository for accessing shared information
        """
        config = config or {}
        super().__init__(agent_executor, role, config, knowledge_repository)
        
        # Research-specific configuration
        self.auto_save_results = config.get("auto_save_results", True)
        self.max_search_results = config.get("max_search_results", 5)
        self.include_sources = config.get("include_sources", True)
        
        logger.debug(f"Initialized ResearchAgent with role: {role}")
    
    def get_capabilities(self) -> List[str]:
        """
        Get the list of capabilities this agent has.
        
        Returns:
            List of capability descriptions
        """
        return [
            "Web search to find relevant information",
            "Information synthesis and summarization",
            "Extraction of key facts and insights",
            "Organization of research findings",
            "Citation and source tracking",
            "Identification of knowledge gaps"
        ]
    
    def research_topic(self, topic: str, depth: str = "medium") -> Dict[str, Any]:
        """
        Conduct comprehensive research on a specific topic.
        
        Args:
            topic: The research topic or question
            depth: Depth of research ("brief", "medium", "comprehensive")
            
        Returns:
            Dictionary with research results
        """
        # Create structured research request
        research_prompt = self._create_research_prompt(topic, depth)
        
        # Execute the research task
        result = self.execute_task(research_prompt)
        
        # Extract and structure the research findings
        structured_results = self._structure_research_results(result, topic)
        
        # Store in knowledge repository if configured
        if self.auto_save_results and self.knowledge_repository:
            self._store_research_results(structured_results, topic)
        
        return structured_results
    
    def _create_research_prompt(self, topic: str, depth: str) -> str:
        """
        Create a detailed research prompt for the given topic.
        
        Args:
            topic: Research topic or question
            depth: Depth of research
            
        Returns:
            Formatted research prompt
        """
        depth_instructions = {
            "brief": "Provide a concise overview with key facts and insights. Keep your research focused on the most important aspects.",
            "medium": "Provide a balanced research report covering main aspects of the topic. Include key facts, some context, and notable insights.",
            "comprehensive": "Conduct thorough research on all aspects of this topic. Include detailed information, historical context, different perspectives, and in-depth analysis."
        }
        
        depth_instruction = depth_instructions.get(depth.lower(), depth_instructions["medium"])
        
        prompt = f"""
        Research Request: {topic}
        
        {depth_instruction}
        
        Please structure your research as follows:
        1. Summary: A concise overview of your findings
        2. Key Facts: The most important facts and data points
        3. Detailed Analysis: In-depth exploration of the topic
        4. Insights & Implications: What these findings mean or suggest
        5. Sources: References to where this information was found (if available)
        
        Use the available search tools to gather accurate and relevant information.
        If certain information isn't available, acknowledge these limitations.
        """
        
        return prompt
    
    def _structure_research_results(self, result: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """
        Structure the raw research results into a consistent format.
        
        Args:
            result: Raw execution result
            topic: Original research topic
            
        Returns:
            Structured research results
        """
        output = result.get("output", "")
        
        # Attempt to parse structured sections from the output
        sections = {
            "summary": "",
            "key_facts": [],
            "detailed_analysis": "",
            "insights": "",
            "sources": []
        }
        
        # Extract sections using simple heuristics
        if "Summary:" in output or "SUMMARY:" in output:
            parts = output.split("Summary:", 1) if "Summary:" in output else output.split("SUMMARY:", 1)
            if len(parts) > 1:
                summary_text = parts[1].split("\n\n", 1)[0].strip()
                sections["summary"] = summary_text
        
        if "Key Facts:" in output or "KEY FACTS:" in output:
            parts = output.split("Key Facts:", 1) if "Key Facts:" in output else output.split("KEY FACTS:", 1)
            if len(parts) > 1:
                facts_text = parts[1].split("\n\n", 1)[0].strip()
                # Split into bullet points or numbered items
                facts = [f.strip() for f in facts_text.split("\n") if f.strip()]
                sections["key_facts"] = facts
        
        if "Detailed Analysis:" in output or "DETAILED ANALYSIS:" in output:
            parts = output.split("Detailed Analysis:", 1) if "Detailed Analysis:" in output else output.split("DETAILED ANALYSIS:", 1)
            if len(parts) > 1:
                analysis_text = parts[1].split("\n\n", 1)[0].strip()
                sections["detailed_analysis"] = analysis_text
        
        if "Insights" in output or "INSIGHTS" in output:
            parts = output.split("Insights", 1) if "Insights" in output else output.split("INSIGHTS", 1)
            if len(parts) > 1:
                insights_text = parts[1].split("\n\n", 1)[0].strip()
                sections["insights"] = insights_text
        
        if "Sources:" in output or "SOURCES:" in output:
            parts = output.split("Sources:", 1) if "Sources:" in output else output.split("SOURCES:", 1)
            if len(parts) > 1:
                sources_text = parts[1].strip()
                # Split into bullet points or numbered items
                sources = [s.strip() for s in sources_text.split("\n") if s.strip()]
                sections["sources"] = sources
        
        # If we couldn't parse structured sections, use the entire output as summary
        if not sections["summary"] and not sections["detailed_analysis"]:
            sections["summary"] = output
        
        # Create the final structured result
        structured_result = {
            "topic": topic,
            "research_data": sections,
            "raw_output": output
        }
        
        return structured_result
    
    def _store_research_results(self, research_results: Dict[str, Any], topic: str):
        """
        Store research results in the knowledge repository.
        
        Args:
            research_results: Structured research results
            topic: Research topic
        """
        if not self.knowledge_repository:
            return
        
        try:
            # Format the content for storage
            content = f"Research on: {topic}\n\n"
            
            # Add summary
            summary = research_results.get("research_data", {}).get("summary", "")
            if summary:
                content += f"Summary:\n{summary}\n\n"
            
            # Add key facts
            key_facts = research_results.get("research_data", {}).get("key_facts", [])
            if key_facts:
                content += "Key Facts:\n"
                for i, fact in enumerate(key_facts, 1):
                    content += f"{i}. {fact}\n"
                content += "\n"
            
            # Add detailed analysis
            analysis = research_results.get("research_data", {}).get("detailed_analysis", "")
            if analysis:
                content += f"Detailed Analysis:\n{analysis}\n\n"
            
            # Add insights
            insights = research_results.get("research_data", {}).get("insights", "")
            if insights:
                content += f"Insights & Implications:\n{insights}\n\n"
            
            # Add sources
            sources = research_results.get("research_data", {}).get("sources", [])
            if sources and self.include_sources:
                content += "Sources:\n"
                for i, source in enumerate(sources, 1):
                    content += f"{i}. {source}\n"
            
            # Store in knowledge repository
            self.knowledge_repository.store_external_knowledge(
                source=f"Research on {topic}",
                content=content,
                metadata={
                    "type": "research",
                    "topic": topic,
                    "agent_role": self.role
                }
            )
            
            logger.info(f"Stored research results for topic: {topic}")
            
        except Exception as e:
            logger.error(f"Error storing research results: {str(e)}")
    
    def find_information(self, query: str, max_results: int = None) -> Dict[str, Any]:
        """
        Find specific information based on a query.
        
        Args:
            query: Information query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        # Use default if not specified
        max_results = max_results or self.max_search_results
        
        # Create search prompt
        search_prompt = f"""
        Information Request: {query}
        
        Please search for this specific information and provide a clear, concise answer.
        If multiple relevant pieces of information are found, include up to {max_results} results.
        
        Include the source of the information when available.
        If the information cannot be found, explain what was searched for and why it might not be available.
        """
        
        # Execute search task
        result = self.execute_task(search_prompt)
        
        # Process and return results
        return {
            "query": query,
            "results": result.get("output", "No results found"),
            "metadata": result.get("metadata", {})
        }
    
    def combine_information(self, sources: List[Dict[str, Any]], query: str = None) -> Dict[str, Any]:
        """
        Combine and synthesize information from multiple sources.
        
        Args:
            sources: List of information sources
            query: Optional context for the synthesis
            
        Returns:
            Dictionary with synthesized information
        """
        # Format sources for the prompt
        formatted_sources = []
        for i, source in enumerate(sources, 1):
            content = source.get("content", "")
            metadata = source.get("metadata", {})
            source_desc = metadata.get("source", f"Source {i}")
            
            formatted_sources.append(f"--- From {source_desc} ---")
            formatted_sources.append(content)
        
        # Create synthesis prompt
        synthesis_prompt = "Synthesize the following information into a coherent, comprehensive response:\n\n"
        synthesis_prompt += "\n\n".join(formatted_sources)
        
        if query:
            synthesis_prompt += f"\n\nThis synthesis should address the following question or topic: {query}"
        
        # Execute synthesis task
        result = self.execute_task(synthesis_prompt)
        
        # Return synthesized information
        return {
            "synthesis": result.get("output", ""),
            "source_count": len(sources),
            "query": query
        }
    
    def _process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the execution result with research-specific enhancements.
        
        Args:
            result: The raw execution result
            
        Returns:
            Processed result
        """
        # Call the base implementation first
        processed = super()._process_result(result)
        
        # Extract any URLs or sources if present in the output
        output = processed.get("output", "")
        sources = []
        
        # Simple extraction of URLs (could be enhanced with regex)
        for line in output.split("\n"):
            if "http://" in line or "https://" in line:
                sources.append(line.strip())
            elif "Source:" in line:
                sources.append(line.strip())
        
        # Add extracted sources to metadata
        if sources and "metadata" in processed:
            processed["metadata"]["extracted_sources"] = sources
        
        return processed
    
    def get_role_description(self) -> str:
        """
        Get a description of this agent's role.
        
        Returns:
            Description of the agent's role
        """
        return (
            f"I am a {self.role} agent specializing in gathering, analyzing, and "
            f"synthesizing information from various sources. I can conduct research "
            f"on topics, find specific information, and combine knowledge from "
            f"multiple sources into coherent insights."
        )
