# src/core/agent.py - SINGLE AGENT IMPLEMENTATION
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum

from .context import AgentContext, AgentResponse
from .component import BaseComponent
from .factory import ComponentFactory
from ..utils.config import AgentConfig

logger = logging.getLogger(__name__)

class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class Agent:
    """
    Single, configurable AI Agent class.
    Replaces both CompleteAgent and IntegratedAgent with component injection.
    """
    
    def __init__(self, agent_id: str = None, config: Union[Dict[str, Any], AgentConfig] = None):
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.config = config if isinstance(config, AgentConfig) else AgentConfig(config or {})
        self.state = AgentState.IDLE
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        
        # Component containers - populated by factory
        self.components: Dict[str, BaseComponent] = {}
        
        # Integration adapters - optional
        self.integrations: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "errors": 0,
            "average_response_time": 0.0,
            "component_stats": {}
        }
        
        # Initialize components based on configuration
        self._factory = ComponentFactory(self.config)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize agent with configured components"""
        if self._initialized:
            return True
            
        try:
            self.update_state(AgentState.PROCESSING)
            
            # Create and initialize components based on config
            self.components = await self._factory.create_components()
            
            # Initialize integrations if specified
            await self._initialize_integrations()
            
            self._initialized = True
            self.update_state(AgentState.IDLE)
            logger.info(f"Agent {self.agent_id} initialized with components: {list(self.components.keys())}")
            return True
            
        except Exception as e:
            self.update_state(AgentState.ERROR)
            logger.error(f"Agent initialization failed: {e}")
            return False
    
    async def _initialize_integrations(self):
        """Initialize optional integrations (LangChain, etc.)"""
        integration_configs = self.config.get_integrations()
        
        for integration_name, integration_config in integration_configs.items():
            if not integration_config.get('enabled', False):
                continue
                
            try:
                if integration_name == 'langchain':
                    await self._setup_langchain_integration(integration_config)
                elif integration_name == 'openai':
                    await self._setup_openai_integration(integration_config)
                # Add other integrations as needed
                
            except Exception as e:
                logger.warning(f"Integration {integration_name} failed to initialize: {e}")
    
    async def _setup_langchain_integration(self, config: Dict[str, Any]):
        """Setup LangChain integration as adapter"""
        from ..integrations.orchestrators.langchain import LangChainAdapter
        
        # Only create if we have required components
        if 'actions' in self.components:
            self.integrations['langchain'] = LangChainAdapter(
                agent=self,
                config=config
            )
            await self.integrations['langchain'].initialize()
    
    async def _setup_openai_integration(self, config: Dict[str, Any]):
        """Setup OpenAI integration"""
        from ..integrations.llm_providers.openai import OpenAIProvider
        
        self.integrations['openai'] = OpenAIProvider(config)
        await self.integrations['openai'].initialize()
    
    async def process_input(self, user_input: str, context: AgentContext = None) -> AgentResponse:
        """Main processing method - orchestrates components"""
        if not self._initialized:
            await self.initialize()
            
        if not context:
            context = AgentContext(user_id="default")
            
        start_time = datetime.now()
        self.update_state(AgentState.PROCESSING)
        
        try:
            response = await self._process_with_components(user_input, context)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(True, processing_time)
            
            self.update_state(AgentState.IDLE)
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(False, processing_time)
            self.update_state(AgentState.ERROR)
            
            return AgentResponse(
                content=f"Error processing request: {str(e)}",
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    async def _process_with_components(self, user_input: str, context: AgentContext) -> AgentResponse:
        """Process using available components in pipeline"""
        processing_data = {
            'user_input': user_input,
            'context': context,
            'metadata': {}
        }
        
        # Component pipeline based on what's available
        pipeline = self._build_processing_pipeline()
        
        for component_name, component in pipeline:
            try:
                result = await component.process(processing_data, context.__dict__)
                processing_data[f'{component_name}_result'] = result
                processing_data['metadata'][component_name] = {
                    'success': True,
                    'processing_time': getattr(result, 'processing_time', 0)
                }
            except Exception as e:
                logger.warning(f"Component {component_name} failed: {e}")
                processing_data['metadata'][component_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return self._build_response(processing_data)
    
    def _build_processing_pipeline(self) -> List[tuple]:
        """Build component processing pipeline based on available components"""
        pipeline = []
        
        # Standard pipeline order
        preferred_order = ['perception', 'memory', 'decision', 'actions', 'learning']
        
        for component_name in preferred_order:
            if component_name in self.components:
                pipeline.append((component_name, self.components[component_name]))
        
        return pipeline
    
    def _build_response(self, processing_data: Dict[str, Any]) -> AgentResponse:
        """Build final response from processing data"""
        # Extract response content from the pipeline
        content = "I'm ready to help!"
        actions_taken = []
        tools_used = []
        confidence = 0.8
        
        # Get response from actions component if available
        if 'actions_result' in processing_data:
            actions_result = processing_data['actions_result']
            content = actions_result.get('response', content)
            actions_taken = actions_result.get('actions_taken', [])
            tools_used = actions_result.get('tools_used', [])
            confidence = actions_result.get('confidence', confidence)
        
        # Or from LangChain integration
        elif 'langchain' in self.integrations:
            # Delegate to LangChain if no core actions available
            langchain_result = processing_data.get('langchain_result', {})
            content = langchain_result.get('response', content)
            tools_used = langchain_result.get('tools_used', [])
        
        return AgentResponse(
            content=content,
            confidence=confidence,
            actions_taken=actions_taken,
            tools_used=tools_used,
            processing_time=sum(
                meta.get('processing_time', 0) 
                for meta in processing_data.get('metadata', {}).values()
                if isinstance(meta, dict)
            ),
            success=True,
            metadata=processing_data.get('metadata', {})
        )
    
    def update_state(self, new_state: AgentState):
        """Update agent state"""
        old_state = self.state
        self.state = new_state
        self.last_active = datetime.now()
        logger.debug(f"Agent {self.agent_id} state: {old_state} -> {new_state}")
    
    def _update_stats(self, success: bool, processing_time: float):
        """Update agent statistics"""
        self.stats['total_interactions'] += 1
        if success:
            self.stats['successful_interactions'] += 1
        else:
            self.stats['errors'] += 1
            
        # Update average response time
        total = self.stats['total_interactions']
        current_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities based on loaded components"""
        capabilities = {
            'agent_id': self.agent_id,
            'components': list(self.components.keys()),
            'integrations': list(self.integrations.keys()),
            'features': []
        }
        
        # Determine features based on components
        if 'perception' in self.components:
            capabilities['features'].append('intent_analysis')
            capabilities['features'].append('entity_extraction')
            
        if 'memory' in self.components:
            capabilities['features'].append('conversation_memory')
            capabilities['features'].append('context_retention')
            
        if 'actions' in self.components:
            capabilities['features'].append('action_execution')
            
        if 'langchain' in self.integrations:
            capabilities['features'].append('tool_orchestration')
            capabilities['features'].append('external_apis')
            
        return capabilities
    
    async def shutdown(self) -> bool:
        """Shutdown agent and cleanup resources"""
        try:
            self.update_state(AgentState.SHUTDOWN)
            
            # Shutdown integrations first
            for integration_name, integration in self.integrations.items():
                if hasattr(integration, 'shutdown'):
                    await integration.shutdown()
            
            # Shutdown components
            for component_name, component in self.components.items():
                if hasattr(component, 'cleanup'):
                    await component.cleanup()
                    
            logger.info(f"Agent {self.agent_id} shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Agent shutdown failed: {e}")
            return False
    
    # Convenience methods for backward compatibility
    @property
    def langchain_agent(self):
        """Backward compatibility property"""
        return self.integrations.get('langchain')
    
    async def process_message(self, message: str, user_id: str = "default"):
        """Backward compatibility method for LangChain interface"""
        context = AgentContext(user_id=user_id)
        response = await self.process_input(message, context)
        
        return {
            'response': response.content,
            'success': response.success,
            'tools_used': response.tools_used,
            'processing_time': response.processing_time
        }