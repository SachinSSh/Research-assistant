import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from ..models.schemas import ContextSummary, FinalBrief, UserHistory
from ..services.llm_service import LLMService
from ..services.storage_service import StorageService


class ContextService:
    def __init__(self, llm_service: LLMService, storage_service: StorageService):
        self.llm_service = llm_service
        self.storage_service = storage_service

    async def get_context_summary(self, user_id: str) -> Optional[ContextSummary]:
        """Generate context summary from user's previous interactions."""
        history = await self.storage_service.get_user_history(user_id)
        
        if not history or not history.briefs:
            return None
        
        recent_cutoff = datetime.utcnow() - timedelta(days=30)
        recent_briefs = [
            brief for brief in history.briefs 
            if brief.generated_at >= recent_cutoff
        ]
        
        if not recent_briefs:
            return None

        topics = [brief.topic for brief in recent_briefs]
        all_findings = []
        for brief in recent_briefs:
            all_findings.extend(brief.key_findings)

        context_prompt = f"""
        Analyze the user's research history and generate a context summary.
        
        Recent Topics: {topics}
        
        Key Findings from Previous Research: {all_findings[:20]}  # Limit to prevent token overflow
        
        Identify:
        1. Recurring themes across topics
        2. Key insights that might be relevant for future research
        3. Areas of expertise or interest
        """

        system_message = """
        You are analyzing a user's research history to create a context summary.
        Focus on identifying patterns, themes, and insights that could inform future research.
        Be concise but comprehensive.
        """

        try:
            context_summary = await self.llm_service.generate_structured(
                context_prompt,
                ContextSummary,
                system_message=system_message,
                use_primary=False
            )
            
            context_summary.user_id = user_id
            context_summary.previous_topics = topics
            context_summary.last_interaction = max(brief.generated_at for brief in recent_briefs)
            context_summary.total_interactions = len(history.briefs)
            
            return context_summary
            
        except Exception as e:
            return ContextSummary(
                user_id=user_id,
                previous_topics=topics[-5:],  
                key_insights=[],
                recurring_themes=[],
                last_interaction=max(brief.generated_at for brief in recent_briefs),
                total_interactions=len(history.briefs)
            )

    async def incorporate_context_into_planning(
        self, 
        topic: str, 
        context: ContextSummary
    ) -> str:
        """Generate context-aware planning prompt."""
        context_prompt = f"""
        Current Research Topic: {topic}
        
        User's Research Context:
        - Previous Topics: {', '.join(context.previous_topics[-5:])}
        - Recurring Themes: {', '.join(context.recurring_themes)}
        - Key Insights: {', '.join(context.key_insights[-10:])}
        - Total Past Research Sessions: {context.total_interactions}
        
        Based on this context, how should we approach researching "{topic}"?
        Consider:
        1. Connections to previous research
        2. Avoiding redundancy with past findings
        3. Building upon established knowledge
        4. Exploring new angles or deeper aspects
        """
        
        system_message = """
        You are helping to plan research that builds upon a user's previous work.
        Provide strategic guidance on how to approach the new topic given their research history.
        Focus on making connections and avoiding redundancy while ensuring comprehensive coverage.
        """
        
        guidance = await self.llm_service.generate_text(
            context_prompt,
            system_message=system_message,
            max_tokens=500
        )
        
        return guidance

    async def save_brief(self, user_id: str, brief: FinalBrief) -> None:
        """Save completed brief to user history."""
        await self.storage_service.save_user_brief(user_id, brief)

    async def get_user_history(self, user_id: str) -> Optional[UserHistory]:
        """Retrieve user's complete research history."""
        return self.storage_service.get_user_history(user_id)