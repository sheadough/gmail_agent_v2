
from pydantic import BaseSettings, Field
from typing import Optional, Dict, Any
import os
from pathlib import Path

class AgentSettings(BaseSettings):
    """Centralized configuration management for AI agents"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # API Keys - These should be set via environment variables or .env file
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    
    # OpenAI Configuration
    openai_model: str = Field(default="gpt-4-1106-preview", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./data/agent_memory.db", 
        env="DATABASE_URL"
    )
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Vector Database Configuration
    vector_db_type: str = Field(default="chroma", env="VECTOR_DB_TYPE")  # chroma, pinecone, faiss
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", 
        env="CHROMA_PERSIST_DIR"
    )
    
    # Agent Configuration
    agent_memory_size: int = Field(default=1000, env="AGENT_MEMORY_SIZE")
    max_conversation_turns: int = Field(default=50, env="MAX_CONVERSATION_TURNS")
    
    # Performance and Rate Limiting
    api_rate_limit: int = Field(default=60, env="API_RATE_LIMIT")  # requests per minute
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")  # seconds
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/agent.log", env="LOG_FILE")
    
    # Web Interface Configuration
    web_host: str = Field(default="0.0.0.0", env="WEB_HOST")
    web_port: int = Field(default=8000, env="WEB_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            Path(self.chroma_persist_directory).parent,
            Path(self.log_file).parent,
            Path("./data"),
            Path("./logs")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary"""
        return {
            "url": self.database_url,
            "echo": self.debug and not self.is_production,
            "future": True
        }

# Create singleton instance
settings = AgentSettings()