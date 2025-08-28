# src/agents/actions/executor.py
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import functools

from ..base.component import BaseComponent

logger = logging.getLogger(__name__)

class ActionStatus(Enum):
    """Status of action execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class ActionPriority(Enum):
    """Priority levels for actions"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class ActionContext:
    """Context for action execution"""
    user_id: str
    session_id: str
    conversation_id: Optional[str] = None
    user_input: str = ""
    perceived_intent: str = "unknown"
    urgency_level: float = 0.0
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActionResult:
    """Result of action execution"""
    action_id: str
    action_type: str
    status: ActionStatus
    result_data: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass  
class ActionDefinition:
    """Definition of an executable action"""
    name: str
    description: str
    executor_func: Callable
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[Callable] = field(default_factory=list)
    postconditions: List[Callable] = field(default_factory=list)
    timeout: float = 30.0
    retry_count: int = 3
    priority: ActionPriority = ActionPriority.NORMAL
    cost_estimate: float = 1.0
    risk_level: float = 0.1
    required_permissions: List[str] = field(default_factory=list)

class ActionQueue:
    """Queue for managing action execution"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.queue: List[tuple] = []  # (priority, timestamp, action_id, action_def, params, context)
        self.running: Dict[str, asyncio.Task] = {}
        self.completed: Dict[str, ActionResult] = {}
        self.lock = asyncio.Lock()
    
    async def enqueue(self, action_def: ActionDefinition, parameters: Dict[str, Any], 
                     context: ActionContext, priority: ActionPriority = None) -> str:
        """Add action to queue"""
        action_id = f"action_{uuid.uuid4().hex[:8]}"
        priority = priority or action_def.priority
        
        async with self.lock:
            # Add to priority queue (higher priority = lower number for heapq)
            queue_item = (
                -priority.value,  # Negative for max-heap behavior
                time.time(),      # Timestamp for FIFO within same priority
                action_id,
                action_def,
                parameters,
                context
            )
            
            # Insert maintaining priority order
            inserted = False
            for i, (p, t, aid, adef, params, ctx) in enumerate(self.queue):
                if -priority.value < p:  # Higher priority
                    self.queue.insert(i, queue_item)
                    inserted = True
                    break
            
            if not inserted:
                self.queue.append(queue_item)
        
        logger.debug(f"Enqueued action {action_id} with priority {priority.name}")
        return action_id
    
    async def process_queue(self):
        """Process actions in the queue"""
        while True:
            try:
                # Clean up completed tasks
                await self._cleanup_completed_tasks()
                
                # Check if we can start new actions
                if len(self.running) < self.max_concurrent and self.queue:
                    async with self.lock:
                        if self.queue:
                            # Get highest priority action
                            priority, timestamp, action_id, action_def, parameters, context = self.queue.pop(0)
                            
                            # Start executing the action
                            task = asyncio.create_task(
                                self._execute_action(action_id, action_def, parameters, context)
                            )
                            self.running[action_id] = task
                            
                            logger.debug(f"Started executing action {action_id}")
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in action queue processing: {e}")
                await asyncio.sleep(1)  # Longer delay on error
    
    async def _cleanup_completed_tasks(self):
        """Clean up completed tasks"""
        completed_ids = []
        
        for action_id, task in self.running.items():
            if task.done():
                completed_ids.append(action_id)
        
        for action_id in completed_ids:
            task = self.running.pop(action_id)
            try:
                result = await task
                self.completed[action_id] = result
            except Exception as e:
                logger.error(f"Action {action_id} failed: {e}")
                # Create error result
                error_result = ActionResult(
                    action_id=action_id,
                    action_type="unknown",
                    status=ActionStatus.FAILED,
                    error_message=str(e)
                )
                self.completed[action_id] = error_result
    
    async def _execute_action(self, action_id: str, action_def: ActionDefinition, 
                             parameters: Dict[str, Any], context: ActionContext) -> ActionResult:
        """Execute a single action"""
        start_time = time.time()
        
        try:
            # Check preconditions
            for precondition in action_def.preconditions:
                if not await self._check_condition(precondition, parameters, context):
                    return ActionResult(
                        action_id=action_id,
                        action_type=action_def.name,
                        status=ActionStatus.FAILED,
                        error_message="Precondition check failed"
                    )
            
            # Execute the action with timeout
            try:
                if asyncio.iscoroutinefunction(action_def.executor_func):
                    result_data = await asyncio.wait_for(
                        action_def.executor_func(parameters, context),
                        timeout=action_def.timeout
                    )
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result_data = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, 
                            functools.partial(action_def.executor_func, parameters, context)
                        ),
                        timeout=action_def.timeout
                    )
                
            except asyncio.TimeoutError:
                return ActionResult(
                    action_id=action_id,
                    action_type=action_def.name,
                    status=ActionStatus.TIMEOUT,
                    error_message=f"Action timed out after {action_def.timeout} seconds",
                    execution_time=time.time() - start_time
                )
            
            # Check postconditions
            for postcondition in action_def.postconditions:
                if not await self._check_condition(postcondition, parameters, context, result_data):
                    return ActionResult(
                        action_id=action_id,
                        action_type=action_def.name,
                        status=ActionStatus.FAILED,
                        error_message="Postcondition check failed",
                        result_data=result_data,
                        execution_time=time.time() - start_time
                    )
            
            # Success
            return ActionResult(
                action_id=action_id,
                action_type=action_def.name,
                status=ActionStatus.COMPLETED,
                result_data=result_data,
                execution_time=time.time() - start_time,
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Action {action_id} execution failed: {e}")
            return ActionResult(
                action_id=action_id,
                action_type=action_def.name,
                status=ActionStatus.FAILED,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _check_condition(self, condition_func: Callable, parameters: Dict[str, Any], 
                              context: ActionContext, result_data: Any = None) -> bool:
        """Check a condition function"""
        try:
            if asyncio.iscoroutinefunction(condition_func):
                return await condition_func(parameters, context, result_data)
            else:
                return condition_func(parameters, context, result_data)
        except Exception as e:
            logger.warning(f"Condition check failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue status"""
        return {
            "queued_actions": len(self.queue),
            "running_actions": len(self.running),
            "completed_actions": len(self.completed),
            "max_concurrent": self.max_concurrent
        }

class ActionRegistry:
    """Registry for available actions"""
    
    def __init__(self):
        self.actions: Dict[str, ActionDefinition] = {}
        self.categories: Dict[str, List[str]] = {}
    
    def register_action(self, action_def: ActionDefinition, category: str = "general"):
        """Register an action"""
        self.actions[action_def.name] = action_def
        
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(action_def.name)
        
        logger.info(f"Registered action: {action_def.name} in category: {category}")
    
    def get_action(self, name: str) -> Optional[ActionDefinition]:
        """Get action by name"""
        return self.actions.get(name)
    
    def list_actions(self, category: str = None) -> List[str]:
        """List available actions"""
        if category:
            return self.categories.get(category, [])
        return list(self.actions.keys())
    
    def search_actions(self, query: str) -> List[str]:
        """Search actions by name or description"""
        query_lower = query.lower()
        matches = []
        
        for name, action_def in self.actions.items():
            if (query_lower in name.lower() or 
                query_lower in action_def.description.lower()):
                matches.append(name)
        
        return matches

# Built-in action implementations
class BuiltinActions:
    """Built-in action implementations"""
    
    @staticmethod
    async def send_response(parameters: Dict[str, Any], context: ActionContext) -> Dict[str, Any]:
        """Send a response to the user"""
        message = parameters.get('message', 'Hello!')
        response_type = parameters.get('type', 'text')
        
        # Simulate response sending
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return {
            'message': message,
            'type': response_type,
            'sent_at': datetime.now().isoformat(),
            'recipient': context.user_id
        }
    
    @staticmethod
    async def log_interaction(parameters: Dict[str, Any], context: ActionContext) -> Dict[str, Any]:
        """Log interaction for analysis"""
        interaction_data = {
            'user_id': context.user_id,
            'session_id': context.session_id,
            'user_input': context.user_input,
            'perceived_intent': context.perceived_intent,
            'timestamp': datetime.now().isoformat(),
            'additional_data': parameters.get('data', {})
        }
        
        # In real implementation, would write to database/file
        logger.info(f"Logged interaction: {interaction_data}")
        
        return {'logged': True, 'log_id': f"log_{uuid.uuid4().hex[:8]}"}
    
    @staticmethod 
    async def escalate_to_human(parameters: Dict[str, Any], context: ActionContext) -> Dict[str, Any]:
        """Escalate conversation to human agent"""
        priority = parameters.get('priority', 'normal')
        reason = parameters.get('reason', 'User request')
        
        # Simulate escalation process
        await asyncio.sleep(0.5)
        
        escalation_data = {
            'escalation_id': f"esc_{uuid.uuid4().hex[:8]}",
            'user_id': context.user_id,
            'priority': priority,
            'reason': reason,
            'context': {
                'user_input': context.user_input,
                'perceived_intent': context.perceived_intent,
                'urgency_level': context.urgency_level
            },
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"Escalated to human: {escalation_data}")
        
        return escalation_data
    
    @staticmethod
    async def store_user_preference(parameters: Dict[str, Any], context: ActionContext) -> Dict[str, Any]:
        """Store user preference"""
        preference_key = parameters.get('key')
        preference_value = parameters.get('value')
        
        if not preference_key:
            raise ValueError("Preference key is required")
        
        # Simulate storage
        await asyncio.sleep(0.1)
        
        return {
            'stored': True,
            'user_id': context.user_id,
            'preference_key': preference_key,
            'preference_value': preference_value,
            'stored_at': datetime.now().isoformat()
        }
    
    @staticmethod
    async def calculate_expression(parameters: Dict[str, Any], context: ActionContext) -> Dict[str, Any]:
        """Calculate mathematical expression"""
        expression = parameters.get('expression', '')
        
        if not expression:
            raise ValueError("Mathematical expression is required")
        
        try:
            # Safe evaluation of mathematical expressions
            import ast
            import operator
            
            # Supported operations
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }
            
            def eval_expr(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise TypeError(f"Unsupported operation: {node}")
            
            result = eval_expr(ast.parse(expression, mode='eval').body)
            
            return {
                'expression': expression,
                'result': result,
                'calculated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise ValueError(f"Invalid mathematical expression: {str(e)}")

class ActionExecutor(BaseComponent):
    """Main action execution engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("action_executor", config)
        
        self.registry = ActionRegistry()
        self.queue = ActionQueue(
            max_concurrent=config.get('max_concurrent_actions', 5) if config else 5
        )
        
        self.execution_stats = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'average_execution_time': 0.0,
            'action_type_stats': {}
        }
        
        # Start queue processing task
        self._queue_task = None
        
        # Register built-in actions
        self._register_builtin_actions()
    
    async def initialize(self) -> bool:
        """Initialize action executor"""
        try:
            # Start queue processing
            self._queue_task = asyncio.create_task(self.queue.process_queue())
            
            self.is_initialized = True
            logger.info("Action executor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize action executor: {e}")
            return False
    
    def _register_builtin_actions(self):
        """Register built-in actions"""
        
        # Response action
        response_action = ActionDefinition(
            name="send_response",
            description="Send a response message to the user",
            executor_func=BuiltinActions.send_response,
            parameters_schema={
                'message': {'type': 'string', 'required': True},
                'type': {'type': 'string', 'default': 'text'}
            },
            timeout=5.0,
            cost_estimate=0.1
        )
        self.registry.register_action(response_action, "communication")
        
        # Logging action
        log_action = ActionDefinition(
            name="log_interaction",
            description="Log interaction data for analysis",
            executor_func=BuiltinActions.log_interaction,
            parameters_schema={
                'data': {'type': 'object', 'required': False}
            },
            timeout=2.0,
            cost_estimate=0.05
        )
        self.registry.register_action(log_action, "logging")
        
        # Escalation action
        escalate_action = ActionDefinition(
            name="escalate_to_human",
            description="Escalate conversation to human agent",
            executor_func=BuiltinActions.escalate_to_human,
            parameters_schema={
                'priority': {'type': 'string', 'default': 'normal'},
                'reason': {'type': 'string', 'required': True}
            },
            timeout=10.0,
            cost_estimate=1.0,
            priority=ActionPriority.HIGH
        )
        self.registry.register_action(escalate_action, "escalation")
        
        # Preference storage action
        preference_action = ActionDefinition(
            name="store_user_preference", 
            description="Store user preference for future interactions",
            executor_func=BuiltinActions.store_user_preference,
            parameters_schema={
                'key': {'type': 'string', 'required': True},
                'value': {'type': 'any', 'required': True}
            },
            timeout=3.0,
            cost_estimate=0.1
        )
        self.registry.register_action(preference_action, "personalization")
        
        # Calculation action
        calc_action = ActionDefinition(
            name="calculate_expression",
            description="Calculate mathematical expressions safely",
            executor_func=BuiltinActions.calculate_expression,
            parameters_schema={
                'expression': {'type': 'string', 'required': True}
            },
            timeout=5.0,
            cost_estimate=0.1
        )
        self.registry.register_action(calc_action, "computation")
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process action execution request"""
        if isinstance(input_data, dict):
            action_name = input_data.get('action')
            parameters = input_data.get('parameters', {})
            priority = ActionPriority(input_data.get('priority', ActionPriority.NORMAL.value))
            action_context = self._create_action_context(context or {})
            
            return await self.execute_action(action_name, parameters, action_context, priority)
        else:
            raise ValueError("Input data must be a dictionary with 'action' and 'parameters' keys")
    
    async def execute_action(self, action_name: str, parameters: Dict[str, Any], 
                           context: ActionContext, priority: ActionPriority = ActionPriority.NORMAL) -> Dict[str, Any]:
        """Execute an action"""
        try:
            # Get action definition
            action_def = self.registry.get_action(action_name)
            if not action_def:
                raise ValueError(f"Unknown action: {action_name}")
            
            # Validate parameters (basic validation)
            self._validate_parameters(parameters, action_def.parameters_schema)
            
            # Add to execution queue
            action_id = await self.queue.enqueue(action_def, parameters, context, priority)
            
            # Wait for completion (with timeout)
            max_wait_time = action_def.timeout + 10.0  # Add buffer time
            wait_start = time.time()
            
            while True:
                if action_id in self.queue.completed:
                    result = self.queue.completed.pop(action_id)
                    
                    # Update statistics
                    self._update_stats(result)
                    
                    return {
                        'action_id': action_id,
                        'action_name': action_name,
                        'status': result.status.value,
                        'result': result.result_data,
                        'error': result.error_message,
                        'execution_time': result.execution_time,
                        'confidence': result.confidence
                    }
                
                # Check timeout
                if time.time() - wait_start > max_wait_time:
                    logger.warning(f"Action {action_id} timed out waiting for completion")
                    return {
                        'action_id': action_id,
                        'action_name': action_name,
                        'status': ActionStatus.TIMEOUT.value,
                        'error': 'Action execution timed out',
                        'execution_time': max_wait_time
                    }
                
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Failed to execute action {action_name}: {e}")
            return {
                'action_name': action_name,
                'status': ActionStatus.FAILED.value,
                'error': str(e),
                'execution_time': 0.0
            }
    
    def _create_action_context(self, context_data: Dict[str, Any]) -> ActionContext:
        """Create ActionContext from dictionary"""
        return ActionContext(
            user_id=context_data.get('user_id', 'unknown'),
            session_id=context_data.get('session_id', f"session_{uuid.uuid4().hex[:8]}"),
            conversation_id=context_data.get('conversation_id'),
            user_input=context_data.get('user_input', ''),
            perceived_intent=context_data.get('perceived_intent', 'unknown'),
            urgency_level=context_data.get('urgency_level', 0.0),
            constraints=context_data.get('constraints', {}),
            metadata=context_data.get('metadata', {})
        )
    
    def _validate_parameters(self, parameters: Dict[str, Any], schema: Dict[str, Any]):
        """Basic parameter validation"""
        for param_name, param_config in schema.items():
            if param_config.get('required', False) and param_name not in parameters:
                raise ValueError(f"Required parameter '{param_name}' is missing")
            
            # Set defaults
            if param_name not in parameters and 'default' in param_config:
                parameters[param_name] = param_config['default']
    
    def _update_stats(self, result: ActionResult):
        """Update execution statistics"""
        self.execution_stats['total_actions'] += 1
        
        if result.status == ActionStatus.COMPLETED:
            self.execution_stats['successful_actions'] += 1
        else:
            self.execution_stats['failed_actions'] += 1
        
        # Update average execution time
        total = self.execution_stats['total_actions']
        current_avg = self.execution_stats['average_execution_time']
        self.execution_stats['average_execution_time'] = (
            (current_avg * (total - 1) + result.execution_time) / total
        )
        
        # Update action type statistics
        action_type = result.action_type
        if action_type not in self.execution_stats['action_type_stats']:
            self.execution_stats['action_type_stats'][action_type] = {
                'count': 0, 'success_count': 0, 'avg_time': 0.0
            }
        
        type_stats = self.execution_stats['action_type_stats'][action_type]
        type_stats['count'] += 1
        
        if result.status == ActionStatus.COMPLETED:
            type_stats['success_count'] += 1
        
        # Update average time for this action type
        type_stats['avg_time'] = (
            (type_stats['avg_time'] * (type_stats['count'] - 1) + result.execution_time) / 
            type_stats['count']
        )
    
    async def execute_parallel_actions(self, actions: List[Dict[str, Any]], 
                                     context: ActionContext, 
                                     max_parallel: int = 3) -> List[Dict[str, Any]]:
        """Execute multiple actions in parallel"""
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_with_semaphore(action_data):
            async with semaphore:
                return await self.execute_action(
                    action_data['action'],
                    action_data.get('parameters', {}),
                    context,
                    ActionPriority(action_data.get('priority', ActionPriority.NORMAL.value))
                )
        
        # Execute all actions concurrently
        tasks = [execute_with_semaphore(action) for action in actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'action_name': actions[i].get('action', 'unknown'),
                    'status': ActionStatus.FAILED.value,
                    'error': str(result),
                    'execution_time': 0.0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def register_custom_action(self, action_def: ActionDefinition, category: str = "custom"):
        """Register a custom action"""
        self.registry.register_action(action_def, category)
    
    def list_available_actions(self, category: str = None) -> List[Dict[str, Any]]:
        """List available actions with details"""
        action_names = self.registry.list_actions(category)
        actions_info = []
        
        for name in action_names:
            action_def = self.registry.get_action(name)
            if action_def:
                actions_info.append({
                    'name': action_def.name,
                    'description': action_def.description,
                    'parameters': action_def.parameters_schema,
                    'timeout': action_def.timeout,
                    'cost_estimate': action_def.cost_estimate,
                    'priority': action_def.priority.name,
                    'risk_level': action_def.risk_level
                })
        
        return actions_info
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        queue_status = self.queue.get_status()
        
        return {
            **self.execution_stats,
            'success_rate': (
                self.execution_stats['successful_actions'] / 
                max(1, self.execution_stats['total_actions']) * 100
            ),
            'queue_status': queue_status,
            'available_actions': len(self.registry.actions)
        }
    
    async def cleanup(self) -> bool:
        """Clean up action executor"""
        try:
            # Cancel queue processing task
            if self._queue_task:
                self._queue_task.cancel()
                try:
                    await self._queue_task
                except asyncio.CancelledError:
                    pass
            
            # Wait for running actions to complete (with timeout)
            if self.queue.running:
                logger.info(f"Waiting for {len(self.queue.running)} running actions to complete...")
                await asyncio.wait_for(
                    asyncio.gather(*self.queue.running.values(), return_exceptions=True),
                    timeout=30.0
                )
            
            self.is_initialized = False
            logger.info("Action executor cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Action executor cleanup failed: {e}")
            return False


# src/agents/actions/decorators.py
"""
Decorators for action functions
"""

import functools
import time
from typing import Callable, Any

def action_timer(func: Callable) -> Callable:
    """Decorator to time action execution"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if isinstance(result, dict):
                result['_execution_time'] = execution_time
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Action {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper

def action_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry failed actions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Action {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        await asyncio.sleep(delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"Action {func.__name__} failed after {max_retries + 1} attempts")
            
            raise last_exception
        
        return wrapper
    return decorator

def requires_permission(permission: str):
    """Decorator to check permissions before action execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(parameters: Dict[str, Any], context: ActionContext):
            # Check if user has required permission
            user_permissions = context.metadata.get('permissions', [])
            if permission not in user_permissions:
                raise PermissionError(f"Action requires permission: {permission}")
            
            return await func(parameters, context)
        
        return wrapper
    return decorator


# src/agents/actions/__init__.py
"""
Action Execution Module

Provides comprehensive action execution capabilities:
- Action registration and management
- Priority-based action queuing
- Parallel and sequential execution
- Built-in actions for common tasks
- Performance monitoring and statistics
- Error handling and retries
"""

from .executor import (
    ActionExecutor,
    ActionDefinition,
    ActionContext,
    ActionResult,
    ActionStatus,
    ActionPriority,
    ActionQueue,
    ActionRegistry,
    BuiltinActions
)

from .decorators import (
    action_timer,
    action_retry,
    requires_permission
)

__all__ = [
    "ActionExecutor",
    "ActionDefinition", 
    "ActionContext",
    "ActionResult",
    "ActionStatus",
    "ActionPriority",
    "ActionQueue",
    "ActionRegistry",
    "BuiltinActions",
    "action_timer",
    "action_retry", 
    "requires_permission"
]