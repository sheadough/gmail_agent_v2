# src/interfaces/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import asyncio
import uvicorn
from datetime import datetime
import logging
import time
import psutil
from contextlib import asynccontextmanager

# Import our components
from src.integrations.langchain.integrated_agent import IntegratedAgent
from src.integrations.databases.vector_store import create_vector_store
from src.utils.metrics import MetricsCollector, metrics_collector
from config.settings import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger(__name__)

# Global agent instance
agent: Optional[IntegratedAgent] = None
security = HTTPBearer(auto_error=False)

# Pydantic models for API
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    user_id: str = Field(default="default", description="User identifier")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    stream: bool = Field(default=False, description="Enable streaming response")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    agent_id: str
    response: str
    analysis: Dict[str, Any]
    tools_used: List[str]
    processing_time: float
    success: bool
    timestamp: str
    user_id: str

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    components: Dict[str, str]

class MetricsResponse(BaseModel):
    """Metrics response model"""
    system_metrics: Dict[str, Any]
    agent_metrics: Dict[str, Any]
    api_metrics: Dict[str, Any]

class MemorySearchRequest(BaseModel):
    """Memory search request model"""
    query: str = Field(..., min_length=1, description="Search query")
    max_results: int = Field(default=5, le=20, description="Maximum results")
    memory_type: Optional[str] = Field(default=None, description="Memory type filter")

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    logger.info("🚀 Starting AI Agent API...")
    await initialize_agent()
    logger.info("✅ AI Agent API ready")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI Agent API...")
    await shutdown_agent()
    logger.info("✅ AI Agent API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="Production-ready AI Agent with LangChain integration, vector memory, and comprehensive monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Collect metrics for all requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    processing_time = time.time() - start_time
    metrics_collector.record_request(
        endpoint=request.url.path,
        response_time=processing_time,
        success=response.status_code < 400
    )
    
    response.headers["X-Process-Time"] = str(processing_time)
    return response

# Authentication dependency (optional)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication dependency"""
    if not credentials and settings.is_production:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required in production"
        )
    return credentials.credentials if credentials else "anonymous"

# Agent initialization
async def initialize_agent():
    """Initialize the AI agent"""
    global agent
    try:
        agent = IntegratedAgent("api_agent_001")
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise

async def shutdown_agent():
    """Shutdown the AI agent"""
    global agent
    if agent:
        try:
            # Perform cleanup operations
            agent.langchain_agent.clear_memory()
            logger.info("Agent shutdown complete")
        except Exception as e:
            logger.error(f"Error during agent shutdown: {e}")

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Agent API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs"
    }

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(get_current_user)
):
    """Main chat endpoint for interacting with the AI agent"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        logger.info(f"Processing chat request from user: {request.user_id}")
        
        # Process the message
        response = await agent.process_input(
            user_input=request.message,
            user_id=request.user_id
        )
        
        # Add background task for additional processing
        background_tasks.add_task(
            log_interaction,
            request.message,
            response,
            request.user_id
        )
        
        if not response['success']:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.get('error', 'Unknown error occurred')
            )
        
        return ChatResponse(
            agent_id=response['agent_id'],
            response=response['response'],
            analysis=response['analysis'],
            tools_used=response['tools_used'],
            processing_time=response['processing_breakdown']['total_time'],
            success=response['success'],
            timestamp=response['timestamp'],
            user_id=response['user_id']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        components = {}
        
        # Check agent status
        if agent:
            components["agent"] = "healthy"
            components["vector_store"] = "healthy"
            components["langchain"] = "healthy"
        else:
            components["agent"] = "unhealthy"
        
        # Check system resources
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        if memory_percent > 90:
            components["memory"] = "critical"
        elif memory_percent > 80:
            components["memory"] = "warning"
        else:
            components["memory"] = "healthy"
        
        if disk_percent > 90:
            components["disk"] = "critical"
        elif disk_percent > 80:
            components["disk"] = "warning"
        else:
            components["disk"] = "healthy"
        
        # Overall status
        overall_status = "healthy"
        if any(status in ["critical", "unhealthy"] for status in components.values()):
            overall_status = "unhealthy"
        elif any(status == "warning" for status in components.values()):
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime_seconds=time.time() - psutil.boot_time(),
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime_seconds=0,
            components={"error": str(e)}
        )

@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics(user: str = Depends(get_current_user)):
    """Get comprehensive system and agent metrics"""
    try:
        # System metrics
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat()
        }
        
        # Agent metrics
        agent_metrics = {}
        if agent:
            agent_metrics = agent.get_comprehensive_stats()
        
        # API metrics
        api_metrics = metrics_collector.get_metrics_summary()
        
        return MetricsResponse(
            system_metrics=system_metrics,
            agent_metrics=agent_metrics,
            api_metrics=api_metrics
        )
        
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )

@app.post("/api/v1/memory/search")
async def search_memory(
    request: MemorySearchRequest,
    user: str = Depends(get_current_user)
):
    """Search agent's vector memory"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # Build metadata filter
        metadata_filter = {}
        if request.memory_type:
            metadata_filter["type"] = request.memory_type
        
        # Search vector store
        results = await agent.vector_store.similarity_search(
            query=request.query,
            k=request.max_results,
            metadata_filter=metadata_filter if metadata_filter else None
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Memory search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Memory search failed"
        )

@app.get("/api/v1/agent/capabilities")
async def get_agent_capabilities():
    """Get agent capabilities and available tools"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        capabilities_text = await agent.langchain_agent.explain_capabilities()
        
        return {
            "capabilities": capabilities_text,
            "available_tools": [tool.name for tool in agent.langchain_agent.tools],
            "tool_descriptions": {
                tool.name: tool.description for tool in agent.langchain_agent.tools
            },
            "features": [
                "Natural language processing",
                "Tool orchestration", 
                "Vector memory",
                "Learning and adaptation",
                "Multi-step reasoning"
            ]
        }
        
    except Exception as e:
        logger.error(f"Capabilities endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get capabilities"
        )

@app.delete("/api/v1/memory/clear")
async def clear_agent_memory(user: str = Depends(get_current_user)):
    """Clear agent's conversation memory"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        agent.langchain_agent.clear_memory()
        
        return {
            "message": "Agent memory cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Memory clear error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear memory"
        )

@app.get("/api/v1/agent/stats")
async def get_agent_stats(user: str = Depends(get_current_user)):
    """Get detailed agent statistics"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        stats = agent.get_comprehensive_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get agent statistics"
        )

# Background task functions
async def log_interaction(message: str, response: Dict[str, Any], user_id: str):
    """Background task to log interactions"""
    try:
        logger.info(f"Interaction logged - User: {user_id}, Tools: {response.get('tools_used', [])}")
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.web_host,
        port=settings.web_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )