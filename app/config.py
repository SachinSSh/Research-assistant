import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    api_host: str = Field(default="127.0.0.1", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    tavily_api_key: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    serp_api_key: Optional[str] = Field(default=None, env="SERP_API_KEY")
    
    database_url: str = Field(default="sqlite:///./research_assistant.db", env="DATABASE_URL")
    
    langchain_tracing_v2: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="research-assistant", env="LANGCHAIN_PROJECT")
    
    max_search_results: int = Field(default=10, env="MAX_SEARCH_RESULTS")
    max_content_length: int = Field(default=8000, env="MAX_CONTENT_LENGTH")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()