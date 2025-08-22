import asyncio
import time
from typing import Dict, Any, Optional, List, Type, TypeVar
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from ..config import settings
from ..utils.monitoring import track_llm_usage

T = TypeVar('T', bound=BaseModel)


class LLMService:
    def __init__(self):
        self.primary_llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            openai_api_key=settings.openai_api_key,
            max_retries=3,
            request_timeout=settings.request_timeout
        )
        
        self.secondary_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            openai_api_key=settings.openai_api_key,
            max_retries=3,
            request_timeout=settings.request_timeout
        )
        
        if settings.anthropic_api_key:
            self.synthesis_llm = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                anthropic_api_key=settings.anthropic_api_key,
                temperature=0.1,
                max_retries=2
            )
        else:
            self.synthesis_llm = self.primary_llm

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @track_llm_usage
    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_message: Optional[str] = None,
        use_primary: bool = True
    ) -> T:
        """Generate structured output using specified schema with retry logic."""
        parser = PydanticOutputParser(pydantic_object=schema)
        
        llm = self.primary_llm if use_primary else self.secondary_llm
        
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        formatted_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"
        messages.append(HumanMessage(content=formatted_prompt))
        
        start_time = time.time()
        response = await llm.ainvoke(messages)
        processing_time = time.time() - start_time
        
        try:
            parsed_output = parser.parse(response.content)
            return parsed_output
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {str(e)}")

    async def generate_text(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        use_synthesis_llm: bool = False
    ) -> str:
        """Generate unstructured text response."""
        llm = self.synthesis_llm if use_synthesis_llm else self.primary_llm
        
        if max_tokens:
            llm = llm.bind(max_tokens=max_tokens)
        
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        response = await llm.ainvoke(messages)
        return response.content

    async def batch_generate_structured(
        self,
        prompts: List[str],
        schema: Type[T],
        system_message: Optional[str] = None,
        max_concurrency: int = 3
    ) -> List[T]:
        """Generate multiple structured outputs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_single(prompt: str) -> T:
            async with semaphore:
                return await self.generate_structured(
                    prompt, schema, system_message, use_primary=False
                )
        
        tasks = [process_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)