# src/agents/learning/system.py
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import numpy as np
from collections import defaultdict, deque
import pickle
from pathlib import Path

from ..base.component import BaseComponent

logger = logging.getLogger(__name__)

class LearningType(Enum):
    """Types of learning"""
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    IMITATION = "imitation"
    FEEDBACK = "feedback"

class FeedbackType(Enum):
    """Types of feedback"""
    EXPLICIT = "explicit"       # Direct user feedback (thumbs up/down)
    IMPLICIT = "implicit"       # Inferred from behavior (completion rate, time spent)
    SYSTEM = "system"          # Internal performance metrics
    CONTEXTUAL = "contextual"  # Environmental feedback

@dataclass
class LearningEpisode:
    """Individual learning episode"""
    episode_id: str
    timestamp: datetime
    context: Dict[str, Any]
    observation: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]
    reward: float
    feedback_type: FeedbackType
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningPattern:
    """Discovered learning pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    conditions: Dict[str, Any]
    actions: List[str]
    success_rate: float
    confidence: float
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None

class ReinforcementLearner:
    """Reinforcement learning component"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table: state-action -> value
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # State-action counts for exploration bonus
        self.visit_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Episode history
        self.episodes: List[LearningEpisode] = []
        self.max_episodes = 10000
    
    def get_state_key(self, observation: Dict[str, Any]) -> str:
        """Convert observation to state key"""
        # Create a simplified state representation
        intent = observation.get('intent', 'unknown')
        urgency = observation.get('urgency', 0.0)
        sentiment = observation.get('sentiment', 0.0)
        
        # Discretize continuous values
        urgency_level = 'low' if urgency < 0.3 else 'medium' if urgency < 0.7 else 'high'
        sentiment_level = 'negative' if sentiment < -0.3 else 'positive' if sentiment > 0.3 else 'neutral'
        
        return f"{intent}_{urgency_level}_{sentiment_level}"
    
    def select_action(self, state_key: str, available_actions: List[str]) -> str:
        """Select action using epsilon-greedy policy"""
        if not available_actions:
            return "default_response"
        
        # Exploration vs exploitation
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(available_actions)
        else:
            # Exploitation: best known action
            action_values = {}
            for action in available_actions:
                base_value = self.q_table[state_key][action]
                
                # Add exploration bonus (UCB-like)
                total_visits = sum(self.visit_counts[state_key].values())
                action_visits = self.visit_counts[state_key][action]
                
                if total_visits > 0 and action_visits > 0:
                    exploration_bonus = math.sqrt(2 * math.log(total_visits) / action_visits)
                    action_values[action] = base_value + 0.1 * exploration_bonus
                else:
                    action_values[action] = base_value + 1.0  # High bonus for unvisited actions
            
            return max(action_values, key=action_values.get)
    
    def update_q_value(self, state_key: str, action: str, reward: float, 
                      next_state_key: str = None, next_actions: List[str] = None):
        """Update Q-value using Q-learning"""
        current_q = self.q_table[state_key][action]
        
        # Calculate max Q-value for next state
        max_next_q = 0.0
        if next_state_key and next_actions:
            next_q_values = [self.q_table[next_state_key][a] for a in next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
        self.visit_counts[state_key][action] += 1
    
    def learn_from_episode(self, episode: LearningEpisode):
        """Learn from a complete episode"""
        state_key = self.get_state_key(episode.observation)
        action = episode.action.get('type', 'unknown')
        reward = episode.reward
        
        # Update Q-value
        self.update_q_value(state_key, action, reward)
        
        # Store episode
        self.episodes.append(episode)
        
        # Manage episode history size
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
    
    def get_action_preferences(self, state_key: str) -> Dict[str, float]:
        """Get action preferences for a given state"""
        return dict(self.q_table[state_key])

class FeedbackAnalyzer:
    """Analyzes user feedback to derive learning signals"""
    
    def __init__(self):
        self.feedback_history: List[Dict[str, Any]] = []
        self.user_patterns: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'satisfaction_scores': [],
            'preferred_response_types': defaultdict(int),
            'interaction_patterns': defaultdict(list)
        })
    
    def process_feedback(self, user_id: str, feedback_data: Dict[str, Any]) -> float:
        """Process feedback and return reward signal"""
        feedback_type = FeedbackType(feedback_data.get('type', 'implicit'))
        
        reward = 0.0
        
        if feedback_type == FeedbackType.EXPLICIT:
            # Direct feedback (thumbs up/down, ratings)
            rating = feedback_data.get('rating', 0)
            if isinstance(rating, bool):
                reward = 1.0 if rating else -1.0
            elif isinstance(rating, (int, float)):
                # Normalize rating to [-1, 1] range
                max_rating = feedback_data.get('max_rating', 5)
                reward = (rating - max_rating/2) / (max_rating/2)
        
        elif feedback_type == FeedbackType.IMPLICIT:
            # Implicit feedback from user behavior
            task_completed = feedback_data.get('task_completed', False)
            response_time = feedback_data.get('response_time', 10.0)
            follow_up_questions = feedback_data.get('follow_up_questions', 0)
            
            # Reward calculation
            reward += 0.5 if task_completed else -0.2
            reward += 0.2 if response_time < 5.0 else -0.1 if response_time > 15.0 else 0
            reward -= follow_up_questions * 0.1
        
        elif feedback_type == FeedbackType.SYSTEM:
            # System performance metrics
            success = feedback_data.get('success', True)
            confidence = feedback_data.get('confidence', 0.5)
            processing_time = feedback_data.get('processing_time', 1.0)
            
            reward += 0.3 if success else -0.5
            reward += (confidence - 0.5) * 0.4
            reward -= min(processing_time / 10.0, 0.2)  # Penalty for slow processing
        
        elif feedback_type == FeedbackType.CONTEXTUAL:
            # Environmental context feedback
            context_match = feedback_data.get('context_match', 0.5)
            appropriateness = feedback_data.get('appropriateness', 0.5)
            
            reward += (context_match - 0.5) * 0.6
            reward += (appropriateness - 0.5) * 0.4
        
        # Store feedback
        feedback_entry = {
            'user_id': user_id,
            'feedback_type': feedback_type.value,
            'feedback_data': feedback_data,
            'calculated_reward': reward,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_history.append(feedback_entry)
        self._update_user_patterns(user_id, feedback_data, reward)
        
        return reward
    
    def _update_user_patterns(self, user_id: str, feedback_data: Dict[str, Any], reward: float):
        """Update user-specific patterns"""
        user_pattern = self.user_patterns[user_id]
        
        # Store satisfaction score
        user_pattern['satisfaction_scores'].append(reward)
        
        # Track preferred response types
        response_type = feedback_data.get('response_type', 'unknown')
        if reward > 0:
            user_pattern['preferred_response_types'][response_type] += 1
        
        # Track interaction patterns
        interaction_context = feedback_data.get('context', {})
        intent = interaction_context.get('intent', 'unknown')
        user_pattern['interaction_patterns'][intent].append({
            'reward': reward,
            'timestamp': datetime.now().isoformat(),
            'context': interaction_context
        })
        
        # Limit history size
        max_history = 1000
        if len(user_pattern['satisfaction_scores']) > max_history:
            user_pattern['satisfaction_scores'] = user_pattern['satisfaction_scores'][-max_history:]
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get learned preferences for a user"""
        if user_id not in self.user_patterns:
            return {}
        
        patterns = self.user_patterns[user_id]
        
        # Calculate average satisfaction
        scores = patterns['satisfaction_scores']
        avg_satisfaction = sum(scores) / len(scores) if scores else 0.0
        
        # Get preferred response types
        response_preferences = dict(patterns['preferred_response_types'])
        
        # Analyze interaction patterns
        intent_performance = {}
        for intent, interactions in patterns['interaction_patterns'].items():
            if interactions:
                avg_reward = sum(i['reward'] for i in interactions) / len(interactions)
                intent_performance[intent] = {
                    'average_reward': avg_reward,
                    'interaction_count': len(interactions)
                }
        
        return {
            'average_satisfaction': avg_satisfaction,
            'preferred_response_types': response_preferences,
            'intent_performance': intent_performance,
            'total_interactions': len(scores)
        }

class PatternDiscovery:
    """Discovers patterns in agent behavior and outcomes"""
    
    def __init__(self, min_pattern_support: int = 5, confidence_threshold: float = 0.7):
        self.min_pattern_support = min_pattern_support
        self.confidence_threshold = confidence_threshold
        self.discovered_patterns: List[LearningPattern] = []
        self.pattern_candidates: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'conditions': {},
            'actions': [],
            'outcomes': [],
            'count': 0
        })
    
    def analyze_episodes(self, episodes: List[LearningEpisode]) -> List[LearningPattern]:
        """Analyze episodes to discover patterns"""
        # Group episodes by similar conditions
        condition_groups = defaultdict(list)
        
        for episode in episodes:
            # Extract key conditions
            condition_key = self._extract_condition_key(episode)
            condition_groups[condition_key].append(episode)
        
        # Find patterns in each group
        new_patterns = []
        
        for condition_key, group_episodes in condition_groups.items():
            if len(group_episodes) >= self.min_pattern_support:
                pattern = self._extract_pattern(condition_key, group_episodes)
                if pattern and pattern.success_rate >= self.confidence_threshold:
                    new_patterns.append(pattern)
        
        # Update discovered patterns
        self.discovered_patterns.extend(new_patterns)
        self._consolidate_patterns()
        
        return new_patterns
    
    def _extract_condition_key(self, episode: LearningEpisode) -> str:
        """Extract key conditions from episode"""
        obs = episode.observation
        
        intent = obs.get('intent', 'unknown')
        urgency = 'high' if obs.get('urgency', 0) > 0.7 else 'low'
        sentiment = 'negative' if obs.get('sentiment', 0) < -0.3 else 'positive' if obs.get('sentiment', 0) > 0.3 else 'neutral'
        
        return f"{intent}_{urgency}_{sentiment}"
    
    def _extract_pattern(self, condition_key: str, episodes: List[LearningEpisode]) -> Optional[LearningPattern]:
        """Extract pattern from grouped episodes"""
        if not episodes:
            return None
        
        # Analyze actions and outcomes
        action_outcomes = defaultdict(list)
        
        for episode in episodes:
            action_type = episode.action.get('type', 'unknown')
            success = episode.outcome.get('success', False)
            reward = episode.reward
            
            action_outcomes[action_type].append({
                'success': success,
                'reward': reward
            })
        
        # Find best performing actions
        best_actions = []
        overall_success_rate = 0.0
        
        for action_type, outcomes in action_outcomes.items():
            success_count = sum(1 for o in outcomes if o['success'])
            success_rate = success_count / len(outcomes)
            avg_reward = sum(o['reward'] for o in outcomes) / len(outcomes)
            
            if success_rate >= self.confidence_threshold and len(outcomes) >= 3:
                best_actions.append(action_type)
                overall_success_rate = max(overall_success_rate, success_rate)
        
        if not best_actions:
            return None
        
        # Create pattern
        conditions = self._parse_condition_key(condition_key)
        
        pattern = LearningPattern(
            pattern_id=f"pattern_{len(self.discovered_patterns)}_{condition_key}",
            pattern_type="behavioral",
            description=f"When {conditions}, prefer actions: {', '.join(best_actions)}",
            conditions=conditions,
            actions=best_actions,
            success_rate=overall_success_rate,
            confidence=min(1.0, len(episodes) / 20.0),  # Confidence based on sample size
            usage_count=0
        )
        
        return pattern
    
    def _parse_condition_key(self, condition_key: str) -> Dict[str, Any]:
        """Parse condition key back to dictionary"""
        parts = condition_key.split('_')
        if len(parts) >= 3:
            return {
                'intent': parts[0],
                'urgency_level': parts[1],
                'sentiment_level': parts[2]
            }
        return {'condition_key': condition_key}
    
    def _consolidate_patterns(self):
        """Remove duplicate or overlapping patterns"""
        # Simple consolidation - remove patterns with identical conditions
        seen_conditions = set()
        consolidated = []
        
        for pattern in sorted(self.discovered_patterns, key=lambda p: p.confidence, reverse=True):
            condition_key = str(pattern.conditions)
            if condition_key not in seen_conditions:
                consolidated.append(pattern)
                seen_conditions.add(condition_key)
        
        self.discovered_patterns = consolidated
    
    def get_applicable_patterns(self, observation: Dict[str, Any]) -> List[LearningPattern]:
        """Get patterns applicable to current observation"""
        applicable = []
        
        for pattern in self.discovered_patterns:
            if self._pattern_matches(pattern.conditions, observation):
                applicable.append(pattern)
        
        return sorted(applicable, key=lambda p: p.confidence, reverse=True)
    
    def _pattern_matches(self, conditions: Dict[str, Any], observation: Dict[str, Any]) -> bool:
        """Check if pattern conditions match observation"""
        for key, expected_value in conditions.items():
            if key == 'intent':
                if observation.get('intent') != expected_value:
                    return False
            elif key == 'urgency_level':
                urgency = observation.get('urgency', 0)
                actual_level = 'high' if urgency > 0.7 else 'low'
                if actual_level != expected_value:
                    return False
            elif key == 'sentiment_level':
                sentiment = observation.get('sentiment', 0)
                if sentiment < -0.3:
                    actual_level = 'negative'
                elif sentiment > 0.3:
                    actual_level = 'positive'
                else:
                    actual_level = 'neutral'
                if actual_level != expected_value:
                    return False
        
        return True

class AdaptationEngine:
    """Manages agent adaptation based on learning"""
    
    def __init__(self):
        self.adaptation_history: List[Dict[str, Any]] = []
        self.adaptation_rules: Dict[str, Callable] = {}
        self.performance_baseline: Dict[str, float] = {
            'success_rate': 0.8,
            'user_satisfaction': 0.7,
            'response_time': 2.0,
            'task_completion_rate': 0.85
        }
        
        # Register default adaptation rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default adaptation rules"""
        self.adaptation_rules['low_success_rate'] = self._adapt_for_low_success
        self.adaptation_rules['slow_response'] = self._adapt_for_slow_response
        self.adaptation_rules['low_satisfaction'] = self._adapt_for_low_satisfaction
    
    def evaluate_performance(self, recent_episodes: List[LearningEpisode]) -> Dict[str, Any]:
        """Evaluate current performance against baseline"""
        if not recent_episodes:
            return {}
        
        # Calculate current metrics
        success_count = sum(1 for e in recent_episodes if e.outcome.get('success', False))
        success_rate = success_count / len(recent_episodes)
        
        avg_satisfaction = sum(e.reward for e in recent_episodes) / len(recent_episodes)
        
        processing_times = [e.outcome.get('processing_time', 0) for e in recent_episodes]
        avg_response_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        completion_count = sum(1 for e in recent_episodes if e.outcome.get('task_completed', False))
        completion_rate = completion_count / len(recent_episodes)
        
        current_performance = {
            'success_rate': success_rate,
            'user_satisfaction': avg_satisfaction,
            'response_time': avg_response_time,
            'task_completion_rate': completion_rate
        }
        
        # Compare with baseline
        performance_gaps = {}
        for metric, current_value in current_performance.items():
            baseline_value = self.performance_baseline[metric]
            
            if metric == 'response_time':
                # Lower is better for response time
                gap = current_value - baseline_value
            else:
                # Higher is better for other metrics
                gap = baseline_value - current_value
            
            performance_gaps[metric] = gap
        
        return {
            'current_performance': current_performance,
            'performance_gaps': performance_gaps,
            'adaptation_needed': any(gap > 0.1 for gap in performance_gaps.values())
        }
    
    def recommend_adaptations(self, performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend adaptations based on performance analysis"""
        recommendations = []
        
        if not performance_analysis.get('adaptation_needed'):
            return recommendations
        
        gaps = performance_analysis.get('performance_gaps', {})
        
        # Check each performance gap
        for metric, gap in gaps.items():
            if gap > 0.1:  # Significant performance gap
                if metric == 'success_rate' and gap > 0.1:
                    recommendations.append({
                        'type': 'low_success_rate',
                        'description': 'Improve action selection and execution',
                        'priority': 'high',
                        'expected_improvement': min(gap * 0.5, 0.2)
                    })
                
                elif metric == 'response_time' and gap > 1.0:
                    recommendations.append({
                        'type': 'slow_response',
                        'description': 'Optimize processing pipeline',
                        'priority': 'medium',
                        'expected_improvement': min(gap * 0.3, 1.0)
                    })
                
                elif metric == 'user_satisfaction' and gap > 0.2:
                    recommendations.append({
                        'type': 'low_satisfaction',
                        'description': 'Improve response quality and personalization',
                        'priority': 'high',
                        'expected_improvement': min(gap * 0.4, 0.3)
                    })
        
        return recommendations
    
    def _adapt_for_low_success(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptation for low success rate"""
        return {
            'action': 'increase_caution',
            'parameters': {
                'increase_validation': True,
                'prefer_conservative_actions': True,
                'increase_confirmation_requests': True
            }
        }
    
    def _adapt_for_slow_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptation for slow response times"""
        return {
            'action': 'optimize_processing',
            'parameters': {
                'reduce_parallel_processing': True,
                'simplify_analysis_pipeline': True,
                'increase_caching': True
            }
        }
    
    def _adapt_for_low_satisfaction(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptation for low user satisfaction"""
        return {
            'action': 'improve_personalization',
            'parameters': {
                'increase_context_awareness': True,
                'enhance_response_quality': True,
                'add_empathy_indicators': True
            }
        }

class LearningSystem(BaseComponent):
    """Main learning and adaptation system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("learning_system", config)
        
        self.config = config or {}
        
        # Initialize learning components
        self.reinforcement_learner = ReinforcementLearner(
            learning_rate=self.config.get('learning_rate', 0.1),
            discount_factor=self.config.get('discount_factor', 0.95),
            epsilon=self.config.get('exploration_rate', 0.1)
        )
        
        self.feedback_analyzer = FeedbackAnalyzer()
        self.pattern_discovery = PatternDiscovery(
            min_pattern_support=self.config.get('min_pattern_support', 5),
            confidence_threshold=self.config.get('pattern_confidence_threshold', 0.7)
        )
        
        self.adaptation_engine = AdaptationEngine()
        
        # Learning history
        self.learning_episodes: deque = deque(maxlen=self.config.get('max_episodes', 10000))
        
        # Performance tracking
        self.learning_stats = {
            'total_episodes': 0,
            'patterns_discovered': 0,
            'adaptations_made': 0,
            'average_reward': 0.0
        }
        
        # Storage paths
        self.storage_path = Path(self.config.get('storage_path', 'data/learning'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize learning system"""
        try:
            # Load existing learning data
            await self._load_learning_data()
            
            self.is_initialized = True
            logger.info("Learning system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize learning system: {e}")
            return False
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process learning input"""
        operation = context.get('operation', 'learn') if context else 'learn'
        
        if operation == 'learn':
            return await self._learn_from_interaction(input_data, context)
        elif operation == 'recommend':
            return await self._get_recommendations(input_data, context)
        elif operation == 'adapt':
            return await self._perform_adaptation(input_data, context)
        else:
            raise ValueError(f"Unknown learning operation: {operation}")
    
    async def learn_from_interaction(self, observation: Dict[str, Any], action: Dict[str, Any], 
                                   outcome: Dict[str, Any], user_feedback: Dict[str, Any] = None) -> Dict[str, Any]:
        """Learn from a complete interaction"""
        try:
            # Process feedback to get reward signal
            reward = 0.0
            if user_feedback:
                user_id = observation.get('user_id', 'unknown')
                reward = self.feedback_analyzer.process_feedback(user_id, user_feedback)
            else:
                # Generate reward from outcome
                reward = self._calculate_outcome_reward(outcome)
            
            # Create learning episode
            episode = LearningEpisode(
                episode_id=f"episode_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                context=observation.get('context', {}),
                observation=observation,
                action=action,
                outcome=outcome,
                reward=reward,
                feedback_type=FeedbackType(user_feedback.get('type', 'system')) if user_feedback else FeedbackType.SYSTEM,
                confidence=outcome.get('confidence', 1.0)
            )
            
            # Add to history
            self.learning_episodes.append(episode)
            
            # Learn from episode
            self.reinforcement_learner.learn_from_episode(episode)
            
            # Update statistics
            self.learning_stats['total_episodes'] += 1
            
            # Update average reward
            total_episodes = self.learning_stats['total_episodes']
            current_avg = self.learning_stats['average_reward']
            self.learning_stats['average_reward'] = (
                (current_avg * (total_episodes - 1) + reward) / total_episodes
            )
            
            # Periodic pattern discovery
            if len(self.learning_episodes) % 50 == 0:
                await self._discover_patterns()
            
            # Periodic adaptation evaluation
            if len(self.learning_episodes) % 100 == 0:
                await self._evaluate_adaptation()
            
            return {
                'episode_id': episode.episode_id,
                'reward': reward,
                'learning_occurred': True,
                'total_episodes': self.learning_stats['total_episodes']
            }
            
        except Exception as e:
            logger.error(f"Learning from interaction failed: {e}")
            return {'learning_occurred': False, 'error': str(e)}
    
    def _calculate_outcome_reward(self, outcome: Dict[str, Any]) -> float:
        """Calculate reward from outcome"""
        reward = 0.0
        
        # Success/failure
        if outcome.get('success', False):
            reward += 0.5
        else:
            reward -= 0.5
        
        # Confidence
        confidence = outcome.get('confidence', 0.5)
        reward += (confidence - 0.5) * 0.3
        
        # Processing efficiency
        processing_time = outcome.get('processing_time', 1.0)
        if processing_time < 1.0:
            reward += 0.2
        elif processing_time > 5.0:
            reward -= 0.2
        
        # User satisfaction indicators
        if outcome.get('task_completed', False):
            reward += 0.3
        
        follow_up_needed = outcome.get('follow_up_needed', False)
        if not follow_up_needed:
            reward += 0.1
        
        return reward
    
    async def _discover_patterns(self):
        """Discover new patterns from recent episodes"""
        try:
            recent_episodes = list(self.learning_episodes)[-200:]  # Last 200 episodes
            new_patterns = self.pattern_discovery.analyze_episodes(recent_episodes)
            
            self.learning_stats['patterns_discovered'] += len(new_patterns)
            
            if new_patterns:
                logger.info(f"Discovered {len(new_patterns)} new learning patterns")
                
        except Exception as e:
            logger.error(f"Pattern discovery failed: {e}")
    
    async def _evaluate_adaptation(self):
        """Evaluate need for adaptation"""
        try:
            recent_episodes = list(self.learning_episodes)[-100:]  # Last 100 episodes
            performance_analysis = self.adaptation_engine.evaluate_performance(recent_episodes)
            
            if performance_analysis.get('adaptation_needed'):
                recommendations = self.adaptation_engine.recommend_adaptations(performance_analysis)
                
                logger.info(f"Adaptation needed. Recommendations: {len(recommendations)}")
                
                # Store recommendations for later use
                self.adaptation_engine.adaptation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'performance_analysis': performance_analysis,
                    'recommendations': recommendations
                })
                
        except Exception as e:
            logger.error(f"Adaptation evaluation failed: {e}")
    
    def get_action_recommendation(self, observation: Dict[str, Any], 
                                available_actions: List[str]) -> Dict[str, Any]:
        """Get action recommendation based on learning"""
        try:
            state_key = self.reinforcement_learner.get_state_key(observation)
            
            # Get reinforcement learning recommendation
            recommended_action = self.reinforcement_learner.select_action(state_key, available_actions)
            action_preferences = self.reinforcement_learner.get_action_preferences(state_key)
            
            # Get pattern-based recommendations
            applicable_patterns = self.pattern_discovery.get_applicable_patterns(observation)
            pattern_recommendations = []
            
            for pattern in applicable_patterns[:3]:  # Top 3 patterns
                pattern_recommendations.append({
                    'pattern_id': pattern.pattern_id,
                    'actions': pattern.actions,
                    'success_rate': pattern.success_rate,
                    'confidence': pattern.confidence
                })
            
            return {
                'recommended_action': recommended_action,
                'action_preferences': action_preferences,
                'pattern_recommendations': pattern_recommendations,
                'state_key': state_key
            }
            
        except Exception as e:
            logger.error(f"Action recommendation failed: {e}")
            return {'recommended_action': available_actions[0] if available_actions else 'default'}
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning system"""
        try:
            insights = {
                'learning_stats': self.learning_stats.copy(),
                'recent_performance': {},
                'discovered_patterns': [],
                'adaptation_history': []
            }
            
            # Recent performance
            if len(self.learning_episodes) >= 20:
                recent_episodes = list(self.learning_episodes)[-20:]
                recent_rewards = [e.reward for e in recent_episodes]
                
                insights['recent_performance'] = {
                    'average_reward': sum(recent_rewards) / len(recent_rewards),
                    'reward_trend': 'improving' if len(recent_rewards) >= 10 and 
                                   sum(recent_rewards[-5:]) / 5 > sum(recent_rewards[-10:-5]) / 5 
                                   else 'stable',
                    'episode_count': len(recent_episodes)
                }
            
            # Top patterns
            top_patterns = sorted(self.pattern_discovery.discovered_patterns, 
                                key=lambda p: p.confidence * p.success_rate, reverse=True)[:5]
            
            insights['discovered_patterns'] = [
                {
                    'pattern_id': p.pattern_id,
                    'description': p.description,
                    'success_rate': p.success_rate,
                    'confidence': p.confidence,
                    'usage_count': p.usage_count
                }
                for p in top_patterns
            ]
            
            # Recent adaptations
            insights['adaptation_history'] = self.adaptation_engine.adaptation_history[-5:]
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return {'error': str(e)}
    
    async def _load_learning_data(self):
        """Load existing learning data from storage"""
        try:
            # Load reinforcement learning data
            rl_path = self.storage_path / 'reinforcement_learning.pkl'
            if rl_path.exists():
                with open(rl_path, 'rb') as f:
                    rl_data = pickle.load(f)
                    self.reinforcement_learner.q_table.update(rl_data.get('q_table', {}))
                    self.reinforcement_learner.visit_counts.update(rl_data.get('visit_counts', {}))
                
                logger.info("Loaded reinforcement learning data")
            
            # Load discovered patterns
            patterns_path = self.storage_path / 'patterns.json'
            if patterns_path.exists():
                with open(patterns_path, 'r') as f:
                    patterns_data = json.load(f)
                    
                for pattern_data in patterns_data.get('patterns', []):
                    pattern = LearningPattern(
                        pattern_id=pattern_data['pattern_id'],
                        pattern_type=pattern_data['pattern_type'],
                        description=pattern_data['description'],
                        conditions=pattern_data['conditions'],
                        actions=pattern_data['actions'],
                        success_rate=pattern_data['success_rate'],
                        confidence=pattern_data['confidence'],
                        usage_count=pattern_data['usage_count'],
                        created_at=datetime.fromisoformat(pattern_data['created_at'])
                    )
                    
                    if pattern_data.get('last_used'):
                        pattern.last_used = datetime.fromisoformat(pattern_data['last_used'])
                    
                    self.pattern_discovery.discovered_patterns.append(pattern)
                
                logger.info(f"Loaded {len(self.pattern_discovery.discovered_patterns)} patterns")
            
        except Exception as e:
            logger.warning(f"Failed to load learning data: {e}")
    
    async def _save_learning_data(self):
        """Save learning data to storage"""
        try:
            # Save reinforcement learning data
            rl_data = {
                'q_table': dict(self.reinforcement_learner.q_table),
                'visit_counts': dict(self.reinforcement_learner.visit_counts)
            }
            
            rl_path = self.storage_path / 'reinforcement_learning.pkl'
            with open(rl_path, 'wb') as f:
                pickle.dump(rl_data, f)
            
            # Save discovered patterns
            patterns_data = {
                'patterns': [
                    {
                        'pattern_id': p.pattern_id,
                        'pattern_type': p.pattern_type,
                        'description': p.description,
                        'conditions': p.conditions,
                        'actions': p.actions,
                        'success_rate': p.success_rate,
                        'confidence': p.confidence,
                        'usage_count': p.usage_count,
                        'created_at': p.created_at.isoformat(),
                        'last_used': p.last_used.isoformat() if p.last_used else None
                    }
                    for p in self.pattern_discovery.discovered_patterns
                ]
            }
            
            patterns_path = self.storage_path / 'patterns.json'
            with open(patterns_path, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            logger.info("Saved learning data successfully")
            
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    async def cleanup(self) -> bool:
        """Clean up learning system"""
        try:
            # Save learning data before cleanup
            await self._save_learning_data()
            
            self.is_initialized = False
            logger.info("Learning system cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Learning system cleanup failed: {e}")
            return False


# src/agents/learning/__init__.py
"""
Learning and Adaptation Module

Provides comprehensive learning capabilities:
- Reinforcement Learning: Q-learning for action optimization
- Feedback Analysis: Multi-modal feedback processing
- Pattern Discovery: Automated behavior pattern extraction
- Adaptation Engine: Performance-based system adaptation
- Learning Insights: Analytics and performance tracking
"""

from .system import (
    LearningSystem,
    LearningType,
    FeedbackType,
    LearningEpisode,
    LearningPattern,
    ReinforcementLearner,
    FeedbackAnalyzer,
    PatternDiscovery,
    AdaptationEngine
)

__all__ = [
    "LearningSystem",
    "LearningType",
    "FeedbackType", 
    "LearningEpisode",
    "LearningPattern",
    "ReinforcementLearner",
    "FeedbackAnalyzer",
    "PatternDiscovery",
    "AdaptationEngine"
]