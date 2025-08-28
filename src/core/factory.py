# src/core/factory.py - Component Factory for Dynamic Loading
from typing import Dict, Any, Optional, Type
import importlib
import logging
from .component import BaseComponent
from ..utils.config import AgentConfig

logger = logging.getLogger(__name__)

class ComponentFactory:
    """Factory for creating and configuring agent components"""
    
    # Component registry - maps component names to module paths
    COMPONENT_REGISTRY = {
        'perception': {
            'basic': 'src.components.perception.basic.BasicPerception',
            'llm_enhanced': 'src.components.perception.llm_enhanced.LLMEnhancedPerception'
        },
        'memory': {
            'local': 'src.components.memory.local.LocalMemory',
            'vector': 'src.components.memory.vector.VectorMemory'
        },
        'decision': {
            'rule_based': 'src.components.decision.rule_based.RuleBasedDecision',
            'llm_based': 'src.components.decision.llm_based.LLMBasedDecision'
        },
        'actions': {
            'basic': 'src.components.actions.basic.BasicActions',
            'tool_calling': 'src.components.actions.tool_calling.ToolCallingActions'
        },
        'learning': {
            'simple': 'src.components.learning.simple.SimpleLearning',
            'rl': 'src.components.learning.rl.ReinforcementLearning'
        }
    }
    
    def __init__(self, config: AgentConfig):
        self.config = config
    
    async def create_components(self) -> Dict[str, BaseComponent]:
        """Create all configured components"""
        components = {}
        component_configs = self.config.get_components()
        
        for component_type, component_config in component_configs.items():
            if component_config is None:
                continue
                
            try:
                component = await self._create_component(component_type, component_config)
                if component:
                    components[component_type] = component
                    
            except Exception as e:
                logger.error(f"Failed to create component {component_type}: {e}")
                # Continue with other components
                
        return components
    
    async def _create_component(self, component_type: str, config: Any) -> Optional[BaseComponent]:
        """Create a single component"""
        if isinstance(config, str):
            # Simple string config - use as implementation name
            implementation = config
            component_config = {}
        elif isinstance(config, dict):
            # Dict config with implementation and settings
            implementation = config.get('implementation', 'basic')
            component_config = config.get('config', {})
        else:
            logger.error(f"Invalid config for component {component_type}: {config}")
            return None
        
        # Get component class
        component_class = self._get_component_class(component_type, implementation)
        if not component_class:
            return None
        
        # Create and initialize component
        try:
            component = component_class(component_config)
            if await component.initialize():
                logger.info(f"Created component: {component_type}.{implementation}")
                return component
            else:
                logger.error(f"Failed to initialize component: {component_type}.{implementation}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating component {component_type}.{implementation}: {e}")
            return None
    
    def _get_component_class(self, component_type: str, implementation: str) -> Optional[Type[BaseComponent]]:
        """Get component class by type and implementation"""
        if component_type not in self.COMPONENT_REGISTRY:
            logger.error(f"Unknown component type: {component_type}")
            return None
            
        if implementation not in self.COMPONENT_REGISTRY[component_type]:
            logger.error(f"Unknown implementation for {component_type}: {implementation}")
            return None
        
        module_path = self.COMPONENT_REGISTRY[component_type][implementation]
        
        try:
            # Parse module path
            module_name, class_name = module_path.rsplit('.', 1)
            
            # Import module and get class
            module = importlib.import_module(module_name)
            component_class = getattr(module, class_name)
            
            # Verify it's a BaseComponent
            if not issubclass(component_class, BaseComponent):
                logger.error(f"Component class {class_name} is not a BaseComponent")
                return None
                
            return component_class
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import component {module_path}: {e}")
            return None
    
    @classmethod
    def register_component(cls, component_type: str, implementation: str, module_path: str):
        """Register a new component implementation"""
        if component_type not in cls.COMPONENT_REGISTRY:
            cls.COMPONENT_REGISTRY[component_type] = {}
            
        cls.COMPONENT_REGISTRY[component_type][implementation] = module_path
        logger.info(f"Registered component: {component_type}.{implementation} -> {module_path}")
    
    @classmethod
    def list_available_components(cls) -> Dict[str, list]:
        """List all available component implementations"""
        return {
            component_type: list(implementations.keys())
            for component_type, implementations in cls.COMPONENT_REGISTRY.items()
        }