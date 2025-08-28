# src/agents/base/agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class AgentContext:
    """Context information for agent operations"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResponse:
    """Standardized agent response format"""
    content: str
    confidence: float = 0.0
    reasoning: Optional[str] = None
    actions_taken: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_id: str = None, config: Dict[str, Any] = None):
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.config = config or {}
        self.state = AgentState.IDLE
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        
        # Core components (to be initialized by subclasses)
        self.perception = None
        self.memory = None
        self.decision_maker = None
        self.action_executor = None
        self.learning_system = None
        
        # Statistics
        self.stats = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "errors": 0,
            "average_response_time": 0.0,
            "uptime_seconds": 0.0
        }
        
        logger.info(f"Initialized agent: {self.agent_id}")
    
    @abstractmethod
    async def process_input(self, user_input: str, context: AgentContext = None) -> AgentResponse:
        """Process user input and return response"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize agent components"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Gracefully shutdown agent"""
        pass
    
    def update_state(self, new_state: AgentState):
        """Update agent state with logging"""
        old_state = self.state
        self.state = new_state
        self.last_active = datetime.now()
        logger.debug(f"Agent {self.agent_id} state changed: {old_state} -> {new_state}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        uptime = (datetime.now() - self.created_at).total_seconds()
        self.stats["uptime_seconds"] = uptime
        
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "stats": self.stats.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform agent health check"""
        components_status = {}
        
        # Check core components
        components_status["perception"] = "healthy" if self.perception else "missing"
        components_status["memory"] = "healthy" if self.memory else "missing"
        components_status["decision_maker"] = "healthy" if self.decision_maker else "missing"
        components_status["action_executor"] = "healthy" if self.action_executor else "missing"
        components_status["learning_system"] = "healthy" if self.learning_system else "missing"
        
        overall_health = "healthy" if all(
            status == "healthy" for status in components_status.values()
        ) else "degraded"
        
        return {
            "agent_id": self.agent_id,
            "overall_health": overall_health,
            "state": self.state.value,
            "components": components_status,
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
            "last_active": self.last_active.isoformat()
        }


# src/agents/base/component.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseComponent(ABC):
    """Base class for all agent components"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        self.error_count = 0
        self.last_error = None
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component"""
        pass
    
    @abstractmethod
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Process input data"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Clean up component resources"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status"""
        return {
            "name": self.name,
            "initialized": self.is_initialized,
            "error_count": self.error_count,
            "last_error": str(self.last_error) if self.last_error else None
        }
    
    async def safe_process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Safely process input with error handling"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            return await self.process(input_data, context)
            
        except Exception as e:
            self.error_count += 1
            self.last_error = e
            logger.error(f"Component {self.name} processing failed: {e}")
            raise


# src/agents/base/__init__.py
"""
Core agent base components

This module provides the fundamental building blocks for AI agents:
- BaseAgent: Abstract base class for all agents
- BaseComponent: Base class for agent components
- AgentContext: Context management for agent operations
- AgentResponse: Standardized response format
"""

from .agent import BaseAgent, AgentState, AgentContext, AgentResponse
from .component import BaseComponent

__all__ = [
    "BaseAgent",
    "AgentState", 
    "AgentContext",
    "AgentResponse",
    "BaseComponent"
]