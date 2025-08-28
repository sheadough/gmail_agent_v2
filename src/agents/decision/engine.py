# src/agents/decision/engine.py
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import math

from ..base.component import BaseComponent

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions the agent can make"""
    RESPONSE = "response"        # How to respond to user
    ACTION = "action"            # What action to take
    TOOL_SELECTION = "tool_selection"  # Which tools to use
    ESCALATION = "escalation"    # Whether to escalate
    LEARNING = "learning"        # What to learn from interaction

@dataclass
class DecisionContext:
    """Context information for decision making"""
    user_input: str
    perception_result: Dict[str, Any]
    relevant_memories: List[Dict[str, Any]] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    user_context: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)

@dataclass
class DecisionOption:
    """Represents a decision option with scoring"""
    option_id: str
    description: str
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    expected_outcome: str = ""
    reasoning: str = ""
    cost: float = 0.0  # Resource cost
    benefit: float = 0.0  # Expected benefit
    risk: float = 0.0  # Associated risk
    priority: float = 0.0  # Overall priority score

@dataclass
class DecisionResult:
    """Result of decision making process"""
    decision_type: DecisionType
    selected_option: DecisionOption
    alternative_options: List[DecisionOption] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    decision_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class DecisionCriteria:
    """Criteria for evaluating decisions"""
    
    def __init__(self):
        self.weights = {
            'relevance': 0.25,      # How relevant to user intent
            'effectiveness': 0.25,   # How well it solves the problem
            'efficiency': 0.15,      # Resource usage
            'safety': 0.20,          # Risk assessment
            'user_satisfaction': 0.15 # Expected user satisfaction
        }
    
    def evaluate_option(self, option: DecisionOption, context: DecisionContext) -> float:
        """Evaluate a decision option against criteria"""
        scores = {}
        
        # Relevance score
        scores['relevance'] = self._calculate_relevance_score(option, context)
        
        # Effectiveness score  
        scores['effectiveness'] = self._calculate_effectiveness_score(option, context)
        
        # Efficiency score
        scores['efficiency'] = self._calculate_efficiency_score(option, context)
        
        # Safety score
        scores['safety'] = self._calculate_safety_score(option, context)
        
        # User satisfaction score
        scores['user_satisfaction'] = self._calculate_satisfaction_score(option, context)
        
        # Calculate weighted score
        total_score = sum(
            score * self.weights[criterion] 
            for criterion, score in scores.items()
        )
        
        return min(1.0, max(0.0, total_score))
    
    def _calculate_relevance_score(self, option: DecisionOption, context: DecisionContext) -> float:
        """Calculate how relevant the option is to user intent"""
        intent = context.perception_result.get('intent', 'other')
        urgency = context.perception_result.get('urgency', 0.0)
        
        # Base relevance by action type
        relevance_map = {
            'response': 0.8,  # Always relevant to provide response
            'search': 0.9 if intent == 'question' else 0.3,
            'calculate': 0.9 if 'calculation' in context.user_input.lower() else 0.2,
            'email': 0.9 if intent == 'request' and 'email' in context.user_input.lower() else 0.1,
            'escalate': urgency * 0.8  # More relevant for urgent requests
        }
        
        base_score = relevance_map.get(option.action_type, 0.5)
        
        # Adjust for urgency
        if urgency > 0.7:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _calculate_effectiveness_score(self, option: DecisionOption, context: DecisionContext) -> float:
        """Calculate expected effectiveness"""
        # Base effectiveness by action type and context
        effectiveness = option.confidence
        
        # Boost for options that match user intent patterns
        intent = context.perception_result.get('intent', 'other')
        if intent == 'question' and option.action_type in ['search', 'response']:
            effectiveness += 0.2
        elif intent == 'request' and option.action_type == 'action':
            effectiveness += 0.2
        
        # Consider available tools
        if option.action_type in context.available_tools:
            effectiveness += 0.1
        
        return min(1.0, effectiveness)
    
    def _calculate_efficiency_score(self, option: DecisionOption, context: DecisionContext) -> float:
        """Calculate resource efficiency"""
        # Lower cost = higher efficiency
        cost_factor = 1.0 - option.cost
        
        # Consider processing time/complexity
        complexity = context.perception_result.get('complexity', 0.5)
        if option.action_type == 'response' and complexity < 0.3:
            cost_factor += 0.2  # Simple responses are efficient
        
        return min(1.0, max(0.0, cost_factor))
    
    def _calculate_safety_score(self, option: DecisionOption, context: DecisionContext) -> float:
        """Calculate safety/risk score"""
        # Base safety (higher = safer)
        safety = 1.0 - option.risk
        
        # Penalize potentially harmful actions
        if option.action_type == 'escalate' and context.perception_result.get('urgency', 0) < 0.5:
            safety -= 0.3  # Don't escalate non-urgent requests
        
        # Reward conservative responses for uncertain situations
        if option.action_type == 'response' and option.confidence < 0.5:
            safety += 0.2
        
        return min(1.0, max(0.0, safety))
    
    def _calculate_satisfaction_score(self, option: DecisionOption, context: DecisionContext) -> float:
        """Calculate expected user satisfaction"""
        sentiment = context.perception_result.get('sentiment', 0.0)
        
        # Base satisfaction
        satisfaction = option.benefit
        
        # Adjust for user sentiment
        if sentiment < -0.5:  # Negative sentiment
            if option.action_type in ['help', 'escalate']:
                satisfaction += 0.3  # Users appreciate help when frustrated
        elif sentiment > 0.5:  # Positive sentiment
            if option.action_type == 'response':
                satisfaction += 0.2  # Continue positive interaction
        
        return min(1.0, satisfaction)

class ResponseDecisionMaker:
    """Specialized decision maker for response generation"""
    
    def __init__(self):
        self.response_templates = {
            'greeting': "Hello! How can I help you today?",
            'question': "Let me help you find that information.",
            'request': "I'll take care of that for you.",
            'complaint': "I understand your frustration. Let me see how I can help resolve this.",
            'compliment': "Thank you! I'm glad I could help.",
            'other': "I'm here to help. Could you tell me more about what you need?"
        }
    
    def generate_response_options(self, context: DecisionContext) -> List[DecisionOption]:
        """Generate response options based on context"""
        options = []
        intent = context.perception_result.get('intent', 'other')
        
        # Direct response option
        direct_response = DecisionOption(
            option_id="direct_response",
            description="Provide direct response",
            action_type="response",
            parameters={
                'template': self.response_templates.get(intent, self.response_templates['other']),
                'personalized': True
            },
            confidence=0.8,
            expected_outcome="User receives immediate response",
            reasoning="Appropriate for most interactions",
            cost=0.1,
            benefit=0.7,
            risk=0.1
        )
        options.append(direct_response)
        
        # Tool-assisted response option
        if context.available_tools:
            tool_response = DecisionOption(
                option_id="tool_assisted_response",
                description="Use tools to enhance response",
                action_type="tool_response",
                parameters={
                    'primary_tools': self._select_relevant_tools(context),
                    'fallback_response': True
                },
                confidence=0.9,
                expected_outcome="Enhanced response with tool assistance",
                reasoning="Tools can provide more accurate/current information",
                cost=0.3,
                benefit=0.9,
                risk=0.2
            )
            options.append(tool_response)
        
        return options
    
    def _select_relevant_tools(self, context: DecisionContext) -> List[str]:
        """Select tools most relevant to the context"""
        intent = context.perception_result.get('intent', 'other')
        user_input = context.user_input.lower()
        
        relevant_tools = []
        
        # Tool selection logic
        if 'calculate' in user_input or 'math' in user_input:
            relevant_tools.append('calculator')
        
        if intent == 'question' or 'search' in user_input:
            relevant_tools.append('web_search')
        
        if 'weather' in user_input:
            relevant_tools.append('get_weather')
        
        if 'email' in user_input or 'send' in user_input:
            relevant_tools.append('send_email')
        
        # Filter by available tools
        available_tools = context.available_tools
        return [tool for tool in relevant_tools if tool in available_tools]

class ActionDecisionMaker:
    """Specialized decision maker for action selection"""
    
    def generate_action_options(self, context: DecisionContext) -> List[DecisionOption]:
        """Generate possible actions based on context"""
        options = []
        intent = context.perception_result.get('intent', 'other')
        urgency = context.perception_result.get('urgency', 0.0)
        
        # Standard response action
        respond_option = DecisionOption(
            option_id="respond",
            description="Provide conversational response",
            action_type="response",
            confidence=0.8,
            cost=0.1,
            benefit=0.6,
            risk=0.1
        )
        options.append(respond_option)
        
        # Information gathering action
        if intent in ['question', 'request']:
            search_option = DecisionOption(
                option_id="gather_info",
                description="Gather information to answer query",
                action_type="search",
                parameters={'search_type': 'comprehensive'},
                confidence=0.7,
                cost=0.3,
                benefit=0.8,
                risk=0.2
            )
            options.append(search_option)
        
        # Problem solving action
        if intent == 'complaint' or urgency > 0.6:
            solve_option = DecisionOption(
                option_id="solve_problem",
                description="Actively solve user's problem",
                action_type="problem_solve",
                parameters={'approach': 'systematic'},
                confidence=0.6,
                cost=0.5,
                benefit=0.9,
                risk=0.3
            )
            options.append(solve_option)
        
        # Escalation action
        if urgency > 0.8 or intent == 'complaint':
            escalate_option = DecisionOption(
                option_id="escalate",
                description="Escalate to human agent",
                action_type="escalate",
                parameters={'priority': 'high' if urgency > 0.8 else 'normal'},
                confidence=0.9,
                cost=0.8,
                benefit=0.7,
                risk=0.4
            )
            options.append(escalate_option)
        
        return options

class DecisionEngine(BaseComponent):
    """Main decision-making engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("decision_engine", config)
        
        self.criteria = DecisionCriteria()
        self.response_maker = ResponseDecisionMaker()
        self.action_maker = ActionDecisionMaker()
        
        # Decision history for learning
        self.decision_history: List[DecisionResult] = []
        self.max_history = config.get('max_decision_history', 1000) if config else 1000
        
        # Performance tracking
        self.decision_stats = {
            'total_decisions': 0,
            'decision_types': {},
            'average_confidence': 0.0,
            'average_decision_time': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize decision engine"""
        try:
            self.is_initialized = True
            logger.info("Decision engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize decision engine: {e}")
            return False
    
    async def process(self, input_data: DecisionContext, context: Optional[Dict[str, Any]] = None) -> DecisionResult:
        """Main decision processing method"""
        decision_type_str = context.get('decision_type', 'response') if context else 'response'
        decision_type = DecisionType(decision_type_str)
        
        start_time = datetime.now()
        
        try:
            # Generate options based on decision type
            if decision_type == DecisionType.RESPONSE:
                options = self.response_maker.generate_response_options(input_data)
            elif decision_type == DecisionType.ACTION:
                options = self.action_maker.generate_action_options(input_data)
            elif decision_type == DecisionType.TOOL_SELECTION:
                options = self._generate_tool_selection_options(input_data)
            else:
                options = self._generate_generic_options(input_data)
            
            # Evaluate and score options
            for option in options:
                option.priority = self.criteria.evaluate_option(option, input_data)
            
            # Select best option
            best_option = max(options, key=lambda x: x.priority)
            alternative_options = sorted(
                [opt for opt in options if opt != best_option],
                key=lambda x: x.priority,
                reverse=True
            )
            
            # Calculate decision time
            decision_time = (datetime.now() - start_time).total_seconds()
            
            # Create decision result
            result = DecisionResult(
                decision_type=decision_type,
                selected_option=best_option,
                alternative_options=alternative_options,
                reasoning=self._generate_decision_reasoning(best_option, alternative_options, input_data),
                confidence=best_option.priority,
                decision_time=decision_time,
                metadata={
                    'options_considered': len(options),
                    'decision_criteria': list(self.criteria.weights.keys())
                }
            )
            
            # Store decision for learning
            self._record_decision(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Decision processing failed: {e}")
            # Return fallback decision
            fallback_option = DecisionOption(
                option_id="fallback",
                description="Provide basic response",
                action_type="response",
                confidence=0.3,
                reasoning="Fallback due to decision processing error"
            )
            
            return DecisionResult(
                decision_type=decision_type,
                selected_option=fallback_option,
                reasoning="Using fallback decision due to processing error",
                confidence=0.3,
                decision_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _generate_tool_selection_options(self, context: DecisionContext) -> List[DecisionOption]:
        """Generate tool selection options"""
        options = []
        available_tools = context.available_tools
        user_input = context.user_input.lower()
        
        # Analyze input for tool requirements
        tool_scores = {}
        
        # Calculator tool
        if any(keyword in user_input for keyword in ['calculate', 'math', 'compute', '+', '-', '*', '/']):
            tool_scores['calculator'] = 0.9
        
        # Web search tool
        if any(keyword in user_input for keyword in ['search', 'find', 'latest', 'current', 'news']):
            tool_scores['web_search'] = 0.8
        
        # Weather tool
        if any(keyword in user_input for keyword in ['weather', 'temperature', 'forecast']):
            tool_scores['get_weather'] = 0.9
        
        # Email tool
        if any(keyword in user_input for keyword in ['email', 'send', 'notify', 'message']):
            tool_scores['send_email'] = 0.8
        
        # Memory search tool
        if any(keyword in user_input for keyword in ['remember', 'recall', 'previous', 'before']):
            tool_scores['memory_search'] = 0.7
        
        # Create options for relevant tools
        for tool, score in tool_scores.items():
            if tool in available_tools:
                option = DecisionOption(
                    option_id=f"use_{tool}",
                    description=f"Use {tool} to assist with request",
                    action_type="tool_use",
                    parameters={'tool': tool},
                    confidence=score,
                    cost=0.2,
                    benefit=score,
                    risk=0.1
                )
                options.append(option)
        
        # Multi-tool option for complex requests
        if len([t for t in tool_scores.keys() if t in available_tools]) > 1:
            multi_tool_option = DecisionOption(
                option_id="multi_tool",
                description="Use multiple tools for comprehensive response",
                action_type="multi_tool",
                parameters={'tools': [t for t in tool_scores.keys() if t in available_tools]},
                confidence=0.7,
                cost=0.5,
                benefit=0.9,
                risk=0.3
            )
            options.append(multi_tool_option)
        
        # No-tool option
        no_tool_option = DecisionOption(
            option_id="no_tools",
            description="Respond without using tools",
            action_type="direct_response",
            confidence=0.6,
            cost=0.1,
            benefit=0.5,
            risk=0.1
        )
        options.append(no_tool_option)
        
        return options
    
    def _generate_generic_options(self, context: DecisionContext) -> List[DecisionOption]:
        """Generate generic options for unknown decision types"""
        return [
            DecisionOption(
                option_id="generic_response",
                description="Provide generic helpful response",
                action_type="response",
                confidence=0.5,
                cost=0.1,
                benefit=0.5,
                risk=0.1
            )
        ]
    
    def _generate_decision_reasoning(self, selected: DecisionOption, alternatives: List[DecisionOption], 
                                  context: DecisionContext) -> str:
        """Generate explanation for decision"""
        reasoning = f"Selected '{selected.description}' (confidence: {selected.confidence:.2f}) because:\n"
        reasoning += f"- Priority score: {selected.priority:.2f}\n"
        reasoning += f"- Expected benefit: {selected.benefit:.2f}\n"
        reasoning += f"- Low risk: {selected.risk:.2f}\n"
        
        if selected.reasoning:
            reasoning += f"- {selected.reasoning}\n"
        
        if alternatives:
            reasoning += f"\nAlternatives considered:\n"
            for alt in alternatives[:2]:  # Show top 2 alternatives
                reasoning += f"- {alt.description} (priority: {alt.priority:.2f})\n"
        
        return reasoning
    
    def _record_decision(self, result: DecisionResult):
        """Record decision for learning and analysis"""
        self.decision_history.append(result)
        
        # Manage history size
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]
        
        # Update statistics
        self.decision_stats['total_decisions'] += 1
        
        decision_type = result.decision_type.value
        if decision_type not in self.decision_stats['decision_types']:
            self.decision_stats['decision_types'][decision_type] = 0
        self.decision_stats['decision_types'][decision_type] += 1
        
        # Update averages
        total = self.decision_stats['total_decisions']
        self.decision_stats['average_confidence'] = (
            (self.decision_stats['average_confidence'] * (total - 1) + result.confidence) / total
        )
        self.decision_stats['average_decision_time'] = (
            (self.decision_stats['average_decision_time'] * (total - 1) + result.decision_time) / total
        )
    
    def make_response_decision(self, context: DecisionContext) -> DecisionResult:
        """Convenience method for response decisions"""
        return asyncio.create_task(
            self.process(context, {'decision_type': 'response'})
        )
    
    def make_action_decision(self, context: DecisionContext) -> DecisionResult:
        """Convenience method for action decisions"""
        return asyncio.create_task(
            self.process(context, {'decision_type': 'action'})
        )
    
    def make_tool_selection_decision(self, context: DecisionContext) -> DecisionResult:
        """Convenience method for tool selection decisions"""
        return asyncio.create_task(
            self.process(context, {'decision_type': 'tool_selection'})
        )
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision-making statistics"""
        return {
            **self.decision_stats,
            'recent_decisions': [
                {
                    'type': d.decision_type.value,
                    'confidence': d.confidence,
                    'decision_time': d.decision_time,
                    'timestamp': d.timestamp.isoformat()
                }
                for d in self.decision_history[-10:]  # Last 10 decisions
            ]
        }
    
    def analyze_decision_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in decision making"""
        if not self.decision_history:
            return {'message': 'No decision history available'}
        
        analysis = {}
        
        # Decision type distribution
        type_counts = {}
        confidence_by_type = {}
        time_by_type = {}
        
        for decision in self.decision_history:
            decision_type = decision.decision_type.value
            
            type_counts[decision_type] = type_counts.get(decision_type, 0) + 1
            
            if decision_type not in confidence_by_type:
                confidence_by_type[decision_type] = []
            confidence_by_type[decision_type].append(decision.confidence)
            
            if decision_type not in time_by_type:
                time_by_type[decision_type] = []
            time_by_type[decision_type].append(decision.decision_time)
        
        analysis['type_distribution'] = type_counts
        analysis['average_confidence_by_type'] = {
            dtype: sum(confidences) / len(confidences)
            for dtype, confidences in confidence_by_type.items()
        }
        analysis['average_time_by_type'] = {
            dtype: sum(times) / len(times)
            for dtype, times in time_by_type.items()
        }
        
        # Recent performance trends
        recent_decisions = self.decision_history[-50:]  # Last 50 decisions
        if recent_decisions:
            recent_confidence = [d.confidence for d in recent_decisions]
            recent_times = [d.decision_time for d in recent_decisions]
            
            analysis['recent_performance'] = {
                'average_confidence': sum(recent_confidence) / len(recent_confidence),
                'average_decision_time': sum(recent_times) / len(recent_times),
                'confidence_trend': self._calculate_trend(recent_confidence),
                'time_trend': self._calculate_trend(recent_times)
            }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    async def cleanup(self) -> bool:
        """Clean up decision engine"""
        try:
            # Save decision history if needed
            logger.info(f"Decision engine processed {self.decision_stats['total_decisions']} decisions")
            
            self.is_initialized = False
            return True
        except Exception as e:
            logger.error(f"Decision engine cleanup failed: {e}")
            return False


# src/agents/decision/strategies.py
"""
Decision-making strategies for different scenarios
"""

from typing import Dict, Any, List
from .engine import DecisionOption, DecisionContext

class ConservativeStrategy:
    """Conservative decision-making strategy"""
    
    @staticmethod
    def adjust_option_scores(options: List[DecisionOption], context: DecisionContext) -> List[DecisionOption]:
        """Adjust option scores to be more conservative"""
        for option in options:
            # Increase weight on safety, reduce risk tolerance
            option.priority = option.priority * (1.0 - option.risk * 0.5)
            
            # Prefer familiar/tested actions
            if option.action_type in ['response', 'search']:
                option.priority *= 1.1
            
        return options

class AggressiveStrategy:
    """Aggressive decision-making strategy"""
    
    @staticmethod 
    def adjust_option_scores(options: List[DecisionOption], context: DecisionContext) -> List[DecisionOption]:
        """Adjust option scores to be more aggressive"""
        for option in options:
            # Increase weight on benefit, reduce concern for cost
            option.priority = option.priority * (1.0 + option.benefit * 0.3)
            option.priority = option.priority * (1.0 - option.cost * 0.2)
            
            # Prefer action-oriented responses
            if option.action_type in ['action', 'tool_use', 'problem_solve']:
                option.priority *= 1.2
        
        return options

class AdaptiveStrategy:
    """Adaptive strategy based on context"""
    
    @staticmethod
    def adjust_option_scores(options: List[DecisionOption], context: DecisionContext) -> List[DecisionOption]:
        """Adjust scores based on situation"""
        urgency = context.perception_result.get('urgency', 0.0)
        sentiment = context.perception_result.get('sentiment', 0.0)
        
        strategy_factor = 1.0
        
        # Be more aggressive for urgent requests
        if urgency > 0.7:
            strategy_factor = 1.2
        # Be more conservative for negative sentiment
        elif sentiment < -0.5:
            strategy_factor = 0.8
        
        for option in options:
            option.priority *= strategy_factor
            
            # Context-specific adjustments
            if urgency > 0.8 and option.action_type == 'escalate':
                option.priority *= 1.3
            elif sentiment < -0.5 and option.action_type == 'response':
                option.priority *= 1.1  # Prioritize thoughtful responses
        
        return options


# src/agents/decision/__init__.py
"""
Decision Making Module

Provides sophisticated decision-making capabilities:
- Multi-criteria decision analysis
- Context-aware option generation
- Risk and benefit assessment
- Decision history and learning
- Multiple decision strategies
"""

from .engine import (
    DecisionEngine,
    DecisionType,
    DecisionContext, 
    DecisionOption,
    DecisionResult,
    DecisionCriteria
)

from .strategies import (
    ConservativeStrategy,
    AggressiveStrategy, 
    AdaptiveStrategy
)

__all__ = [
    "DecisionEngine",
    "DecisionType",
    "DecisionContext",
    "DecisionOption", 
    "DecisionResult",
    "DecisionCriteria",
    "ConservativeStrategy",
    "AggressiveStrategy",
    "AdaptiveStrategy"
]