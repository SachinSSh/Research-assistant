import time
import asyncio
import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from ..models.state import GraphState, NodeOutput
from ..models.schemas import (
    ResearchPlan, SearchResult, SourceSummary, FinalBrief, 
    Reference, ContextSummary
)
from ..services.llm_service import LLMService
from ..services.search_service import SearchService
from ..services.context_service import ContextService
from ..utils.monitoring import track_node_execution


class ResearchNodes:
    def __init__(
        self, 
        llm_service: LLMService,
        search_service: SearchService,
        context_service: ContextService
    ):
        self.llm_service = llm_service
        self.search_service = search_service
        self.context_service = context_service

    @track_node_execution
    async def context_summarization_node(self, state: GraphState) -> GraphState:
        """Summarize user's previous research context if this is a follow-up query."""
        try:
            if not state["request"].follow_up:
                state["context_summary"] = None
                return state

            context_summary = await self.context_service.get_context_summary(
                state["request"].user_id
            )
            
            state["context_summary"] = context_summary
            state["messages"].append(
                HumanMessage(content=f"Context summary generated for user {state['request'].user_id}")
            )
            
        except Exception as e:
            state["errors"].append(f"Context summarization failed: {str(e)}")
            state["context_summary"] = None
        
        return state

    @track_node_execution
    async def planning_node(self, state: GraphState) -> GraphState:
        """Create a research plan based on the topic and context."""
        try:
            topic = state["request"].topic
            depth = state["request"].depth
            context = state.get("context_summary")
            planning_prompt = f"""
            Research Topic: {topic}
            Research Depth: {depth.name} ({depth.value}/3)
            
            Create a comprehensive research plan that includes:
            1. Specific search queries to find relevant information
            2. Expected number of sources to analyze
            3. Key focus areas to investigate
            4. Estimated duration for the research
            """
            
            if context:
                context_guidance = await self.context_service.incorporate_context_into_planning(
                    topic, context
                )
                planning_prompt += f"\n\nContext Guidance:\n{context_guidance}"
            
            system_message = """
            You are a research planning expert. Create detailed, actionable research plans
            that will lead to comprehensive and well-sourced research briefs.
            Consider the research depth level when determining scope and thoroughness.
            """
            
            research_plan = await self.llm_service.generate_structured(
                planning_prompt,
                ResearchPlan,
                system_message=system_message
            )
            
            research_plan.query = topic
            state["research_plan"] = research_plan
            state["messages"].append(
                HumanMessage(content=f"Research plan created with {len(research_plan.search_queries)} search queries")
            )
            
        except Exception as e:
            state["errors"].append(f"Planning failed: {str(e)}")
            state["research_plan"] = ResearchPlan(
                query=topic,
                search_queries=[topic],
                expected_sources=5,
                focus_areas=[topic],
                estimated_duration=300
            )
        
        return state

    @track_node_execution
    async def search_node(self, state: GraphState) -> GraphState:
        """Execute search queries to find relevant sources."""
        try:
            plan = state["research_plan"]
            all_results = []
            
            search_tasks = [
                self.search_service.search_web(query, max_results=5)
                for query in plan.search_queries[:5]  
            ]
            
            search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for i, results in enumerate(search_results_list):
                if isinstance(results, Exception):
                    state["errors"].append(f"Search {i+1} failed: {str(results)}")
                    continue
                all_results.extend(results)
            
            unique_results = {}
            for result in all_results:
                if result.url not in unique_results:
                    unique_results[result.url] = result
            
            sorted_results = sorted(
                unique_results.values(),
                key=lambda x: x.relevance_score,
                reverse=True
            )
            
            max_results = min(plan.expected_sources, 15)  # Cap at 15 sources
            state["search_results"] = sorted_results[:max_results]
            
            state["messages"].append(
                HumanMessage(content=f"Found {len(state['search_results'])} search results")
            )
            
        except Exception as e:
            state["errors"].append(f"Search failed: {str(e)}")
            state["search_results"] = []
        
        return state

    @track_node_execution
    async def content_fetching_node(self, state: GraphState) -> GraphState:
        """Fetch content from search result URLs."""
        try:
            search_results = state["search_results"]
            urls = [result.url for result in search_results]
            
            content_dict = await self.search_service.batch_fetch_content(urls, max_concurrency=5)
            
            state["fetched_content"] = content_dict
            
            successful_fetches = len([c for c in content_dict.values() if not c.startswith("Error") and not c.startswith("Failed")])
            state["messages"].append(
                HumanMessage(content=f"Successfully fetched content from {successful_fetches}/{len(urls)} sources")
            )
            
        except Exception as e:
            state["errors"].append(f"Content fetching failed: {str(e)}")
            state["fetched_content"] = {}
        
        return state

    @track_node_execution
    async def source_summarization_node(self, state: GraphState) -> GraphState:
        """Summarize content from each source."""
        try:
            search_results = state["search_results"]
            fetched_content = state["fetched_content"]
            
            summarization_tasks = []
            
            for result in search_results:
                content = fetched_content.get(result.url, "")
                if content and not content.startswith(("Error", "Failed")) and len(content) > 100:
                    
                    summary_prompt = f"""
                    Source: {result.title}
                    URL: {result.url}
                    Content: {content[:3000]}  # Limit content length
                    
                    Analyze this source and extract:
                    1. Key points relevant to the research topic
                    2. Important insights or findings
                    3. Credibility indicators (author expertise, publication quality, etc.)
                    4. How this source contributes to understanding the topic
                    """
                    
                    summarization_tasks.append(
                        self._summarize_single_source(
                            summary_prompt, result.url, result.title, content
                        )
                    )
            
            batch_size = 5
            all_summaries = []
            
            for i in range(0, len(summarization_tasks), batch_size):
                batch = summarization_tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, SourceSummary):
                        all_summaries.append(result)
                    elif isinstance(result, Exception):
                        state["errors"].append(f"Source summarization error: {str(result)}")
            
            state["source_summaries"] = all_summaries
            state["messages"].append(
                HumanMessage(content=f"Summarized {len(all_summaries)} sources")
            )
            
        except Exception as e:
            state["errors"].append(f"Source summarization failed: {str(e)}")
            state["source_summaries"] = []
        
        return state

    async def _summarize_single_source(
        self, 
        prompt: str, 
        url: str, 
        title: str, 
        content: str
    ) -> SourceSummary:
        """Summarize a single source."""
        system_message = """
        You are analyzing a source for research purposes. Extract key information,
        assess credibility, and determine relevance to the research topic.
        Be concise but comprehensive in your analysis.
        """
        
        try:
            summary = await self.llm_service.generate_structured(
                prompt,
                SourceSummary,
                system_message=system_message,
                use_primary=False
            )
            
            summary.url = url
            summary.title = title
            summary.word_count = len(content.split())
            
            return summary
            
        except Exception as e:
            return SourceSummary(
                url=url,
                title=title,
                content_snippet=content[:500],
                key_points=["Content analysis failed"],
                relevance_score=0.5,
                credibility_score=0.5,
                word_count=len(content.split())
            )

    @track_node_execution
    async def synthesis_node(self, state: GraphState) -> GraphState:
        """Synthesize all source summaries into a final research brief."""
        try:
            topic = state["request"].topic
            source_summaries = state["source_summaries"]
            context = state.get("context_summary")
            
            if not source_summaries:
                raise ValueError("No source summaries available for synthesis")
            
            sources_text = "\n\n".join([
                f"Source {i+1}: {summary.title}\n"
                f"Key Points: {', '.join(summary.key_points)}\n"
                f"Relevance: {summary.relevance_score:.2f}\n"
                f"Snippet: {summary.content_snippet}"
                for i, summary in enumerate(source_summaries)
            ])
            
            synthesis_prompt = f"""
            Research Topic: {topic}
            
            Source Summaries:
            {sources_text}
            
            Create a comprehensive research brief that:
            1. Provides a clear summary of the topic
            2. Identifies key findings from the research
            3. Offers detailed analysis and insights
            4. Maintains high confidence in the conclusions
            5. Properly references all sources
            
            Ensure the brief is well-structured, informative, and actionable.
            """
            
            if context:
                synthesis_prompt += f"""
                
                User Context: This user has previously researched: {', '.join(context.previous_topics[-3:])}
                Consider building upon their existing knowledge while avoiding redundancy.
                """
            
            system_message = """
            You are an expert research analyst creating comprehensive research briefs.
            Synthesize information from multiple sources into coherent, actionable insights.
            Maintain objectivity while highlighting the most important findings.
            Ensure all claims are supported by the source material.
            """
            
            final_brief = await self.llm_service.generate_structured(
                synthesis_prompt,
                FinalBrief,
                system_message=system_message,
                use_primary=True
            )
            
            final_brief.topic = topic
            final_brief.processing_time = time.time() - state["processing_start"]
            
            references = []
            for summary in source_summaries:
                if summary.relevance_score > 0.6:  
                    references.append(Reference(
                        title=summary.title,
                        url=summary.url,
                        excerpt=summary.content_snippet[:200]
                    ))
            
            final_brief.references = references[:10]  
            
            avg_relevance = sum(s.relevance_score for s in source_summaries) / len(source_summaries)
            avg_credibility = sum(s.credibility_score for s in source_summaries) / len(source_summaries)
            final_brief.confidence_score = (avg_relevance + avg_credibility) / 2
            
            state["final_brief"] = final_brief
            state["messages"].append(
                HumanMessage(content=f"Research brief synthesized with {len(references)} references")
            )
            
        except Exception as e:
            state["errors"].append(f"Synthesis failed: {str(e)}")
            state["final_brief"] = FinalBrief(
                topic=topic,
                summary="Research synthesis failed due to technical issues.",
                key_findings=["Unable to complete research due to errors"],
                detailed_analysis="The research process encountered technical difficulties and could not be completed successfully.",
                references=[],
                confidence_score=0.0,
                processing_time=time.time() - state["processing_start"],
                token_usage=state.get("token_usage", {})
            )
        
        return state

    @track_node_execution
    async def post_processing_node(self, state: GraphState) -> GraphState:
        """Final processing and cleanup."""
        try:
            if state["final_brief"] and state["request"].user_id:
                await self.context_service.save_brief(
                    state["request"].user_id,
                    state["final_brief"]
                )
            
            if state["final_brief"]:
                state["final_brief"].token_usage = state.get("token_usage", {})
            
            if "processing_start" in state:
                total_time = time.time() - state["processing_start"]
                state["processing_time"] = total_time
            
            state.pop("fetched_content", None)  
            
            state["messages"].append(
                HumanMessage(content="Research brief completed and saved")
            )
            
        except Exception as e:
            state["errors"].append(f"Post-processing failed: {str(e)}")
        
        return state

    async def cleanup_node(self, state: GraphState) -> GraphState:
        """Clean up resources and finalize state."""
        try:
            if hasattr(self.search_service, 'close'):
                await self.search_service.close()
            
            if state.get("final_brief"):
                brief = state["final_brief"]
                processing_time = state.get("processing_time", 0)
                num_sources = len(brief.references)
                
                state["messages"].append(
                    HumanMessage(content=f"Research completed in {processing_time:.2f}s with {num_sources} sources")
                )
            
            state["completed"] = True
            
        except Exception as e:
            state["errors"].append(f"Cleanup failed: {str(e)}")
            
        return state

    def get_error_summary(self, state: GraphState) -> str:
        """Generate a summary of all errors encountered during processing."""
        errors = state.get("errors", [])
        if not errors:
            return "No errors encountered during processing."
        
        error_summary = f"Encountered {len(errors)} errors during processing:\n"
        for i, error in enumerate(errors, 1):
            error_summary += f"{i}. {error}\n"
        
        return error_summary

    def get_processing_stats(self, state: GraphState) -> Dict[str, Any]:
        """Generate processing statistics."""
        stats = {
            "total_time": state.get("processing_time", 0),
            "search_queries": len(state.get("research_plan", {}).get("search_queries", [])),
            "search_results": len(state.get("search_results", [])),
            "source_summaries": len(state.get("source_summaries", [])),
            "final_references": len(state.get("final_brief", {}).get("references", [])),
            "errors": len(state.get("errors", [])),
            "token_usage": state.get("token_usage", {}),
        }
        
        if state.get("final_brief"):
            stats["confidence_score"] = state["final_brief"].confidence_score
        
        return stats