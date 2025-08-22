import time
import functools
from typing import Dict, Any, Callable
from langsmith import Client
from ..config import settings


langsmith_client = None
if settings.langchain_api_key:
    langsmith_client = Client(api_key=settings.langchain_api_key)


def setup_tracing(trace_id: str):
    """Setup tracing for the current execution."""
    pass


def track_llm_usage(func: Callable) -> Callable:
    """Decorator to track LLM usage and costs."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            if langsmith_client:
                langsmith_client.create_run(
                    name=f"llm_call_{func.__name__}",
                    run_type="llm",
                    start_time=start_time,
                    end_time=time.time(),
                    extra={"processing_time": processing_time}
                )
            
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            
            if langsmith_client:
                langsmith_client.create_run(
                    name=f"llm_call_{func.__name__}_failed",
                    run_type="llm",
                    start_time=start_time,
                    end_time=time.time(),
                    error=str(e),
                    extra={"processing_time": processing_time}
                )
            
            raise
    
    return wrapper


def track_api_call(func: Callable) -> Callable:
    """Decorator to track external API calls."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            print(f"API call {func.__name__} completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"API call {func.__name__} failed after {processing_time:.2f}s: {str(e)}")
            raise
    
    return wrapper


def track_node_execution(func: Callable) -> Callable:
    """Decorator to track LangGraph node execution."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        node_name = func.__name__.replace("_node", "")
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            print(f"Node {node_name} completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Node {node_name} failed after {processing_time:.2f}s: {str(e)}")
            raise
    
    return wrapper