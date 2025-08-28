# src/agents/core/complete_agent.py
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import logging
import uuid

from ..base.agent import BaseAgent, AgentState, AgentContext, AgentResponse
from ..perception.processor import PerceptionProcessor
from ..memory.manager import MemoryManager
from ..decision.engine import DecisionEngine, DecisionContext
from ..actions.executor import ActionExecutor, ActionContext
from ..learning.system import LearningSystem

# Import integrations
from ...integrations.langchain.agent_executor import LangChainAgent
from ...integrations.databases.vector_store import create_vector_store

logger = logging.getLogger(__name__)

class CompleteAgent(BaseAgent):
    """Complete AI agent with all core components integrated"""
    
    def __init__(self, agent_id: str = None, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        # Initialize core components
        self.perception = None
        self.memory = None
        self.decision_maker = None
        self.action_executor = None
        self.learning_system = None
        
        # Initialize LangChain integration
        self.langchain_agent = None
        self.vector_store = None
        
        # Component configuration
        self.component_config = {
            'perception': config.get('perception', {}) if config else {},
            'memory': config.get('memory', {}) if config else {},
            'decision': config.get('decision', {}) if config else {},
            'actions': config.get('actions', {}) if config else {},
            'learning': config.get('learning', {}) if config else {},
            'langchain': config.get('langchain', {}) if config else {},
            'vector_store': config.get('vector_store', {}) if config else {}
        }
        
        # Integration stats
        self.integration_stats = {
            'core_component_cycles': 0,
            'langchain_interactions': 0,
            'memory_operations': 0,
            'learning_episodes': 0,
            'total_processing_time': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize all agent components"""
        try:
            self.update_state(AgentState.PROCESSING)
            logger.info(f"Initializing complete agent: {self.agent_id}")
            
            # Initialize vector store first (used by other components)
            await self._initialize_vector_store()
            
            # Initialize core components
            await self._initialize_perception()
            await self._initialize_memory()
            await self._initialize_decision_engine()
            await self._initialize_action_executor()
            await self._initialize_learning_system()
            
            # Initialize LangChain integration
            await self._initialize_langchain()
            
            self.update_state(AgentState.IDLE)
            logger.info(f"Agent {self.agent_id} initialized successfully with all components")
            return True
            
        except Exception as e:
            self.update_state(AgentState.ERROR)
            logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            return False
    
    async def _initialize_vector_store(self):
        """Initialize vector store for memory"""
        try:
            collection_name = f"{self.agent_id}_memory"
            self.vector_store = create_vector_store(collection_name)
            logger.info("Vector store initialized")
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {e}")
    
    async def _initialize_perception(self):
        """Initialize perception component"""
        perception_config = self.component_config['perception']
        self.perception = PerceptionProcessor(perception_config)
        
        success = await self.perception.initialize()
        if not success:
            raise Exception("Perception component initialization failed")
        
        logger.info("Perception component initialized")
    
    async def _initialize_memory(self):
        """Initialize memory component"""
        memory_config = self.component_config['memory'].copy()
        
        # Add vector store to memory config
        if self.vector_store:
            memory_config['use_vector_store'] = True
            memory_config['vector_store'] = self.vector_store
        
        self.memory = MemoryManager(memory_config)
        
        success = await self.memory.initialize()
        if not success:
            raise Exception("Memory component initialization failed")
        
        logger.info("Memory component initialized")
    
    async def _initialize_decision_engine(self):
        """Initialize decision engine"""
        decision_config = self.component_config['decision']
        self.decision_maker = DecisionEngine(decision_config)
        
        success = await self.decision_maker.initialize()
        if not success:
            raise Exception("Decision engine initialization failed")
        
        logger.info("Decision engine initialized")
    
    async def _initialize_action_executor(self):
        """Initialize action executor"""
        action_config = self.component_config['actions']
        self.action_executor = ActionExecutor(action_config)
        
        success = await self.action_executor.initialize()
        if not success:
            raise Exception("Action executor initialization failed")
        
        logger.info("Action executor initialized")
    
    async def _initialize_learning_system(self):
        """Initialize learning system"""
        learning_config = self.component_config['learning']
        self.learning_system = LearningSystem(learning_config)
        
        success = await self.learning_system.initialize()
        if not success:
            raise Exception("Learning system initialization failed")
        
        logger.info("Learning system initialized")
    
    async def _initialize_langchain(self):
        """Initialize LangChain integration"""
        try:
            langchain_config = self.component_config['langchain']
            
            # Create custom tools if specified
            custom_tools = langchain_config.get('custom_tools')
            
            self.langchain_agent = LangChainAgent(
                vector_store=self.vector_store,
                custom_tools=custom_tools
            )
            
            logger.info("LangChain integration initialized")
            
        except Exception as e:
            logger.warning(f"LangChain initialization failed: {e}")
            # Continue without LangChain if initialization fails
    
    async def process_input(self, user_input: str, context: AgentContext = None) -> AgentResponse:
        """Main processing method that integrates all components"""
        if not context:
            context = AgentContext(user_id="default")
        
        start_time = datetime.now()
        self.update_state(AgentState.PROCESSING)
        
        try:
            # Step 1: Perception - Analyze input
            logger.debug(f"Step 1: Perception analysis for: {user_input[:100]}...")
            perception_result = await self.perception.process(user_input)
            
            # Step 2: Memory - Retrieve relevant context
            logger.debug("Step 2: Memory retrieval")
            memory_context = await self._retrieve_memory_context(user_input, context, perception_result)
            
            # Step 3: Decision Making - Determine best approach
            logger.debug("Step 3: Decision making")
            decision_context = self._create_decision_context(
                user_input, perception_result, memory_context, context
            )
            
            # Get action recommendation from learning system
            available_actions = self._get_available_actions()
            learning_recommendation = self.learning_system.get_action_recommendation(
                perception_result.__dict__, available_actions
            )
            
            # Make decision
            decision_result = await self.decision_maker.process(
                decision_context, {'decision_type': 'response'}
            )
            
            # Step 4: Action Execution - Execute chosen action
            logger.debug("Step 4: Action execution")
            
            # Choose execution strategy based on decision
            if self.langchain_agent and decision_result.selected_option.action_type in ['tool_use', 'tool_response']:
                # Use LangChain for tool-based responses
                execution_result = await self._execute_with_langchain(user_input, context)
            else:
                # Use core action executor
                execution_result = await self._execute_with_core_actions(
                    decision_result, context, perception_result
                )
            
            # Step 5: Learning - Learn from interaction
            logger.debug("Step 5: Learning from interaction")
            await self._learn_from_interaction(
                perception_result, decision_result, execution_result, context
            )
            
            # Step 6: Memory Storage - Store interaction
            logger.debug("Step 6: Memory storage")
            await self._store_interaction_memory(
                user_input, perception_result, decision_result, execution_result, context
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self.integration_stats['total_processing_time'] += processing_time
            self.integration_stats['core_component_cycles'] += 1
            
            # Update agent statistics
            self.stats['total_interactions'] += 1
            if execution_result.get('success', False):
                self.stats['successful_interactions'] += 1
            
            # Create comprehensive response
            response = AgentResponse(
                content=execution_result.get('response', ''),
                confidence=execution_result.get('confidence', 0.5),
                reasoning=decision_result.reasoning,
                actions_taken=execution_result.get('actions_taken', []),
                tools_used=execution_result.get('tools_used', []),
                processing_time=processing_time,
                success=execution_result.get('success', False),
                metadata={
                    'perception': {
                        'intent': perception_result.intent,
                        'sentiment': perception_result.sentiment,
                        'urgency': perception_result.urgency,
                        'entities': len(perception_result.entities)
                    },
                    'decision': {
                        'selected_option': decision_result.selected_option.option_id,
                        'confidence': decision_result.confidence,
                        'alternatives': len(decision_result.alternative_options)
                    },
                    'learning': {
                        'recommendation_used': learning_recommendation.get('recommended_action'),
                        'patterns_applied': len(learning_recommendation.get('pattern_recommendations', []))
                    },
                    'memory': {
                        'context_retrieved': len(memory_context.get('working_memory', [])),
                        'total_memories': memory_context.get('memory_stats', {}).get('total_items', 0)
                    }
                }
            )
            
            self.update_state(AgentState.IDLE)
            return response
            
        except Exception as e:
            self.update_state(AgentState.ERROR)
            logger.error(f"Agent processing failed: {e}")
            
            # Return error response
            processing_time = (datetime.now() - start_time).total_seconds()
            return AgentResponse(
                content="I apologize, but I encountered an error processing your request. Please try again.",
                confidence=0.1,
                reasoning=f"Processing error: {str(e)}",
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _retrieve_memory_context(self, user_input: str, context: AgentContext, 
                                     perception_result) -> Dict[str, Any]:
        """Retrieve relevant memory context"""
        try:
            # Search across all memory types
            memory_results = await self.memory.retrieve_memory(
                user_input, 
                {
                    'user_id': context.user_id,
                    'intent': perception_result.intent,
                    'metadata_filter': {'user_id': context.user_id}
                }
            )
            
            # Get memory statistics
            memory_stats = self.memory.get_memory_stats()
            
            self.integration_stats['memory_operations'] += 1
            
            return {
                **memory_results,
                'memory_stats': memory_stats
            }
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return {'working_memory': [], 'episodic_memory': [], 'semantic_memory': [], 'vector_memory': []}
    
    def _create_decision_context(self, user_input: str, perception_result, 
                               memory_context: Dict[str, Any], context: AgentContext) -> DecisionContext:
        """Create decision context from gathered information"""
        
        # Extract relevant memories
        relevant_memories = []
        for memory_type, memories in memory_context.items():
            if memory_type != 'memory_stats' and isinstance(memories, list):
                relevant_memories.extend(memories[:3])  # Top 3 from each type
        
        return DecisionContext(
            user_input=user_input,
            perception_result=perception_result.__dict__,
            relevant_memories=[m.__dict__ if hasattr(m, '__dict__') else m for m in relevant_memories],
            available_tools=self._get_available_tools(),
            user_context={
                'user_id': context.user_id,
                'session_id': context.session_id,
                'preferences': context.preferences,
                'conversation_history': context.conversation_history
            },
            constraints=context.metadata.get('constraints', {}),
            goals=['provide_helpful_response', 'maintain_user_satisfaction']
        )
    
    def _get_available_actions(self) -> List[str]:
        """Get list of available actions"""
        core_actions = self.action_executor.registry.list_actions()
        
        langchain_tools = []
        if self.langchain_agent:
            langchain_tools = [tool.name for tool in self.langchain_agent.tools]
        
        return core_actions + langchain_tools
    
    def _get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        if self.langchain_agent:
            return [tool.name for tool in self.langchain_agent.tools]
        return []
    
    async def _execute_with_langchain(self, user_input: str, context: AgentContext) -> Dict[str, Any]:
        """Execute using LangChain agent"""
        try:
            result = await self.langchain_agent.process_message(user_input, context.user_id)
            
            self.integration_stats['langchain_interactions'] += 1
            
            return {
                'response': result['response'],
                'success': result['success'],
                'tools_used': result['tools_used'],
                'actions_taken': ['langchain_processing'],
                'confidence': 0.8,  # Default confidence for LangChain responses
                'processing_time': result['processing_time'],
                'metadata': {
                    'langchain_details': result.get('tool_details', []),
                    'intermediate_steps': result.get('intermediate_steps', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"LangChain execution failed: {e}")
            return {
                'response': 'I encountered an issue processing your request with my tools.',
                'success': False,
                'tools_used': [],
                'actions_taken': ['langchain_error'],
                'confidence': 0.1,
                'error': str(e)
            }
    
    async def _execute_with_core_actions(self, decision_result, context: AgentContext, 
                                       perception_result) -> Dict[str, Any]:
        """Execute using core action system"""
        try:
            selected_option = decision_result.selected_option
            
            # Create action context
            action_context = ActionContext(
                user_id=context.user_id,
                session_id=context.session_id,
                user_input=context.user_context.get('current_input', ''),
                perceived_intent=perception_result.intent,
                urgency_level=perception_result.urgency,
                constraints=context.metadata.get('constraints', {}),
                metadata=context.metadata
            )
            
            # Execute the selected action
            action_result = await self.action_executor.execute_action(
                selected_option.action_type,
                selected_option.parameters,
                action_context
            )
            
            return {
                'response': action_result.get('result', {}).get('message', 'Action completed.'),
                'success': action_result.get('status') == 'completed',
                'tools_used': [],
                'actions_taken': [selected_option.action_type],
                'confidence': action_result.get('confidence', selected_option.confidence),
                'processing_time': action_result.get('execution_time', 0),
                'metadata': {
                    'action_id': action_result.get('action_id'),
                    'action_details': action_result
                }
            }
            
        except Exception as e:
            logger.error(f"Core action execution failed: {e}")
            return {
                'response': 'I was unable to complete the requested action.',
                'success': False,
                'tools_used': [],
                'actions_taken': ['action_error'],
                'confidence': 0.1,
                'error': str(e)
            }
    
    async def _learn_from_interaction(self, perception_result, decision_result, 
                                    execution_result: Dict[str, Any], context: AgentContext):
        """Learn from the completed interaction"""
        try:
            # Prepare learning data
            observation = {
                'user_id': context.user_id,
                'intent': perception_result.intent,
                'sentiment': perception_result.sentiment,
                'urgency': perception_result.urgency,
                'complexity': perception_result.complexity,
                'entities': len(perception_result.entities),
                'context': context.user_context
            }
            
            action = {
                'type': decision_result.selected_option.action_type,
                'confidence': decision_result.confidence,
                'parameters': decision_result.selected_option.parameters,
                'reasoning': decision_result.reasoning
            }
            
            outcome = {
                'success': execution_result.get('success', False),
                'confidence': execution_result.get('confidence', 0.5),
                'processing_time': execution_result.get('processing_time', 0),
                'tools_used': execution_result.get('tools_used', []),
                'actions_taken': execution_result.get('actions_taken', [])
            }
            
            # Learn from interaction
            learning_result = await self.learning_system.learn_from_interaction(
                observation, action, outcome
            )
            
            self.integration_stats['learning_episodes'] += 1
            
            logger.debug(f"Learning completed: {learning_result.get('episode_id')}")
            
        except Exception as e:
            logger.error(f"Learning from interaction failed: {e}")
    
    async def _store_interaction_memory(self, user_input: str, perception_result, 
                                      decision_result, execution_result: Dict[str, Any], 
                                      context: AgentContext):
        """Store interaction in memory"""
        try:
            # Create interaction record
            interaction_data = {
                'user_input': user_input,
                'user_id': context.user_id,
                'session_id': context.session_id,
                'timestamp': datetime.now().isoformat(),
                'perception': {
                    'intent': perception_result.intent,
                    'sentiment': perception_result.sentiment,
                    'urgency': perception_result.urgency,
                    'entities': [e for e in perception_result.entities if len(str(e)) < 100]  # Limit size
                },
                'decision': {
                    'selected_action': decision_result.selected_option.action_type,
                    'confidence': decision_result.confidence
                },
                'outcome': {
                    'success': execution_result.get('success', False),
                    'response_preview': execution_result.get('response', '')[:200]  # First 200 chars
                },
                'performance_metrics': {
                    'processing_time': execution_result.get('processing_time', 0),
                    'tools_used_count': len(execution_result.get('tools_used', []))
                }
            }
            
            # Store in episodic memory
            await self.memory.store_memory(
                interaction_data,
                {
                    'memory_type': 'episodic',
                    'importance': self._calculate_interaction_importance(perception_result, execution_result),
                    'tags': ['interaction', perception_result.intent, context.user_id],
                    'metadata': {
                        'user_id': context.user_id,
                        'success': execution_result.get('success', False)
                    }
                }
            )
            
            logger.debug("Interaction stored in memory")
            
        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
    
    def _calculate_interaction_importance(self, perception_result, execution_result: Dict[str, Any]) -> float:
        """Calculate importance score for memory storage"""
        importance = 0.5  # Base importance
        
        # Increase importance for high urgency
        importance += perception_result.urgency * 0.2
        
        # Increase importance for successful interactions
        if execution_result.get('success', False):
            importance += 0.2
        
        # Increase importance for strong sentiment
        importance += abs(perception_result.sentiment) * 0.1
        
        # Increase importance for complex interactions
        importance += perception_result.complexity * 0.1
        
        # Increase importance if tools were used
        if execution_result.get('tools_used'):
            importance += 0.1
        
        return min(1.0, importance)
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        try:
            status = {
                'agent_info': {
                    'agent_id': self.agent_id,
                    'state': self.state.value,
                    'created_at': self.created_at.isoformat(),
                    'last_active': self.last_active.isoformat(),
                    'uptime_seconds': (datetime.now() - self.created_at).total_seconds()
                },
                'integration_stats': self.integration_stats.copy(),
                'component_health': {},
                'performance_metrics': {}
            }
            
            # Component health checks
            components = {
                'perception': self.perception,
                'memory': self.memory,
                'decision_maker': self.decision_maker,
                'action_executor': self.action_executor,
                'learning_system': self.learning_system,
                'langchain_agent': self.langchain_agent
            }
            
            for name, component in components.items():
                if component:
                    if hasattr(component, 'health_check'):
                        health = await component.health_check()
                    elif hasattr(component, 'is_initialized'):
                        health = {'status': 'healthy' if component.is_initialized else 'unhealthy'}
                    else:
                        health = {'status': 'unknown'}
                    
                    status['component_health'][name] = health
                else:
                    status['component_health'][name] = {'status': 'not_initialized'}
            
            # Performance metrics
            if self.stats['total_interactions'] > 0:
                status['performance_metrics'] = {
                    'success_rate': (self.stats['successful_interactions'] / self.stats['total_interactions']) * 100,
                    'average_processing_time': self.integration_stats['total_processing_time'] / self.stats['total_interactions'],
                    'interactions_per_hour': self.stats['total_interactions'] / max(1, (datetime.now() - self.created_at).total_seconds() / 3600)
                }
            
            # Component-specific stats
            if self.learning_system:
                status['learning_insights'] = self.learning_system.get_learning_insights()
            
            if self.memory:
                status['memory_stats'] = self.memory.get_memory_stats()
            
            if self.action_executor:
                status['action_stats'] = self.action_executor.get_execution_stats()
            
            if self.decision_maker:
                status['decision_stats'] = self.decision_maker.get_decision_stats()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive status: {e}")
            return {'error': str(e), 'agent_id': self.agent_id}
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown agent and all components"""
        try:
            logger.info(f"Shutting down agent: {self.agent_id}")
            self.update_state(AgentState.SHUTDOWN)
            
            # Shutdown components in reverse order
            components = [
                ('learning_system', self.learning_system),
                ('action_executor', self.action_executor),
                ('decision_maker', self.decision_maker),
                ('memory', self.memory),
                ('perception', self.perception)
            ]
            
            for name, component in components:
                if component and hasattr(component, 'cleanup'):
                    try:
                        success = await component.cleanup()
                        if success:
                            logger.info(f"Component {name} shut down successfully")
                        else:
                            logger.warning(f"Component {name} shutdown returned failure")
                    except Exception as e:
                        logger.error(f"Error shutting down component {name}: {e}")
            
            # Clear references
            self.perception = None
            self.memory = None
            self.decision_maker = None
            self.action_executor = None
            self.learning_system = None
            self.langchain_agent = None
            self.vector_store = None
            
            logger.info(f"Agent {self.agent_id} shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Agent shutdown failed: {e}")
            return False


# src/agents/core/__init__.py
"""
Complete AI Agent Core

Brings together all agent components into a unified, production-ready system:
- Perception: Advanced input analysis with LLM enhancement
- Memory: Multi-layered memory with vector search
- Decision Making: Multi-criteria decision analysis
- Action Execution: Priority-based action management
- Learning: Reinforcement learning and pattern discovery
- LangChain Integration: Tool orchestration and reasoning
"""

from .complete_agent import CompleteAgent
from ..base.agent import BaseAgent, AgentState, AgentContext, AgentResponse

__all__ = [
    "CompleteAgent",
    "BaseAgent", 
    "AgentState",
    "AgentContext",
    "AgentResponse"
]