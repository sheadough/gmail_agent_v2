# src/agents/perception/processor.py
from typing import Dict, Any, List, Optional
import re
import asyncio
from datetime import datetime
import logging
from dataclasses import dataclass, field
import json

from ..base.component import BaseComponent

logger = logging.getLogger(__name__)

@dataclass
class PerceptionResult:
    """Result of perception processing"""
    raw_input: str
    processed_text: str
    intent: str
    confidence: float
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: float = 0.0
    emotion: str = "neutral"
    urgency: float = 0.0
    complexity: float = 0.0
    topics: List[str] = field(default_factory=list)
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle special characters
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    @staticmethod
    def extract_metadata(text: str) -> Dict[str, Any]:
        """Extract basic metadata from text"""
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len([s for s in text.split('.') if s.strip()]),
            "has_questions": '?' in text,
            "has_exclamations": '!' in text,
            "has_urls": bool(re.search(r'https?://\S+', text)),
            "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            "has_numbers": bool(re.search(r'\d+', text)),
            "has_dates": bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text))
        }

class IntentClassifier:
    """Intent classification for user input"""
    
    # Intent patterns (can be enhanced with ML models)
    INTENT_PATTERNS = {
        'question': [
            r'\b(what|how|when|where|why|who|which|can you|could you|would you)\b',
            r'\?',
            r'\b(explain|tell me|show me)\b'
        ],
        'request': [
            r'\b(please|could you|can you|would you|help me)\b',
            r'\b(do|make|create|generate|find|search)\b',
            r'\b(I need|I want|I would like)\b'
        ],
        'complaint': [
            r'\b(problem|issue|error|wrong|not working|broken|frustrated)\b',
            r'\b(hate|dislike|terrible|awful|bad)\b',
            r'\b(fix|solve|resolve)\b'
        ],
        'compliment': [
            r'\b(thank|thanks|appreciate|great|excellent|amazing|wonderful)\b',
            r'\b(good job|well done|impressive|helpful)\b',
            r'\b(love|like|enjoy)\b'
        ],
        'greeting': [
            r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
            r'\b(how are you|nice to meet|goodbye|bye|see you)\b'
        ],
        'information': [
            r'\b(I am|I have|my name is|I live|I work)\b',
            r'\b(here is|this is|let me tell you)\b'
        ],
        'urgent': [
            r'\b(urgent|emergency|asap|immediately|critical|important)\b',
            r'\b(quick|fast|hurry|rush)\b',
            r'!!+'
        ]
    }
    
    def classify_intent(self, text: str) -> tuple[str, float]:
        """Classify intent with confidence score"""
        text_lower = text.lower()
        scores = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            # Normalize score
            scores[intent] = score / len(patterns)
        
        if not scores or max(scores.values()) == 0:
            return 'other', 0.1
        
        best_intent = max(scores, key=scores.get)
        confidence = min(0.9, scores[best_intent] * 0.5)  # Cap at 0.9
        
        return best_intent, confidence

class EntityExtractor:
    """Extract entities from text"""
    
    ENTITY_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'url': r'https?://\S+',
        'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        'time': r'\b\d{1,2}:\d{2}(?:\s?[AaPp][Mm])?\b',
        'money': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
        'percentage': r'\d+(?:\.\d+)?%',
        'number': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
    }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        entities = []
        
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': entity_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8  # Rule-based confidence
                })
        
        return entities

class SentimentAnalyzer:
    """Simple rule-based sentiment analysis"""
    
    POSITIVE_WORDS = {
        'love', 'like', 'enjoy', 'great', 'excellent', 'amazing', 'wonderful',
        'good', 'nice', 'happy', 'pleased', 'satisfied', 'thank', 'thanks',
        'appreciate', 'helpful', 'perfect', 'awesome', 'fantastic'
    }
    
    NEGATIVE_WORDS = {
        'hate', 'dislike', 'terrible', 'awful', 'bad', 'horrible', 'frustrated',
        'angry', 'annoyed', 'disappointed', 'problem', 'issue', 'error',
        'wrong', 'broken', 'useless', 'stupid', 'worst'
    }
    
    def analyze_sentiment(self, text: str) -> tuple[float, str]:
        """Analyze sentiment returning score (-1 to 1) and emotion"""
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in self.POSITIVE_WORDS)
        negative_count = sum(1 for word in words if word in self.NEGATIVE_WORDS)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0, "neutral"
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / len(words)
        sentiment_score = max(-1.0, min(1.0, sentiment_score * 5))  # Scale and clamp
        
        # Determine emotion
        if sentiment_score > 0.3:
            emotion = "positive"
        elif sentiment_score < -0.3:
            emotion = "negative"
        else:
            emotion = "neutral"
        
        return sentiment_score, emotion

class PerceptionProcessor(BaseComponent):
    """Main perception processor that orchestrates all analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("perception_processor", config)
        
        # Initialize processors
        self.preprocessor = TextPreprocessor()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Enhanced analysis flag
        self.use_llm_enhancement = config.get('use_llm_enhancement', False)
        
    async def initialize(self) -> bool:
        """Initialize perception processor"""
        try:
            # Initialize LLM enhancement if configured
            if self.use_llm_enhancement:
                # Import LLM-enhanced perception
                from ...integrations.openai.enhanced_perception import LLMEnhancedPerceptionModule
                self.llm_enhancer = LLMEnhancedPerceptionModule()
            
            self.is_initialized = True
            logger.info("Perception processor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize perception processor: {e}")
            return False
    
    async def process(self, input_data: str, context: Optional[Dict[str, Any]] = None) -> PerceptionResult:
        """Process input text through perception pipeline"""
        try:
            # Step 1: Preprocess text
            cleaned_text = self.preprocessor.clean_text(input_data)
            metadata = self.preprocessor.extract_metadata(cleaned_text)
            
            # Step 2: Intent classification
            intent, intent_confidence = self.intent_classifier.classify_intent(cleaned_text)
            
            # Step 3: Entity extraction
            entities = self.entity_extractor.extract_entities(cleaned_text)
            
            # Step 4: Sentiment analysis
            sentiment_score, emotion = self.sentiment_analyzer.analyze_sentiment(cleaned_text)
            
            # Step 5: Calculate complexity and urgency
            complexity = self._calculate_complexity(cleaned_text, entities)
            urgency = self._calculate_urgency(intent, cleaned_text, sentiment_score)
            
            # Step 6: Topic identification (basic)
            topics = self._identify_topics(cleaned_text, entities)
            
            # Step 7: LLM enhancement (if enabled)
            if self.use_llm_enhancement and hasattr(self, 'llm_enhancer'):
                try:
                    enhanced_result = await self.llm_enhancer.enhanced_perceive(cleaned_text)
                    
                    # Merge LLM results with rule-based results
                    intent = enhanced_result.get('intent', intent)
                    intent_confidence = max(intent_confidence, enhanced_result.get('intent_confidence', 0))
                    entities.extend(enhanced_result.get('entities', []))
                    sentiment_score = enhanced_result.get('sentiment', sentiment_score)
                    emotion = enhanced_result.get('emotional_tone', emotion)
                    topics = enhanced_result.get('topics', topics)
                    
                except Exception as e:
                    logger.warning(f"LLM enhancement failed, using rule-based results: {e}")
            
            # Create result
            result = PerceptionResult(
                raw_input=input_data,
                processed_text=cleaned_text,
                intent=intent,
                confidence=intent_confidence,
                entities=entities,
                sentiment=sentiment_score,
                emotion=emotion,
                urgency=urgency,
                complexity=complexity,
                topics=topics,
                metadata=metadata
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Perception processing failed: {e}")
            # Return minimal result on error
            return PerceptionResult(
                raw_input=input_data,
                processed_text=input_data,
                intent='other',
                confidence=0.1
            )
    
    def _calculate_complexity(self, text: str, entities: List[Dict[str, Any]]) -> float:
        """Calculate text complexity score"""
        # Basic complexity metrics
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        # Factors
        word_count_factor = min(1.0, len(words) / 50)  # Normalize to 50 words
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        word_length_factor = min(1.0, avg_word_length / 7)  # Normalize to 7 chars
        entity_factor = min(1.0, len(entities) / 5)  # Normalize to 5 entities
        
        complexity = (word_count_factor + word_length_factor + entity_factor) / 3
        return complexity
    
    def _calculate_urgency(self, intent: str, text: str, sentiment: float) -> float:
        """Calculate urgency score"""
        urgency = 0.0
        
        # Intent-based urgency
        if intent == 'urgent':
            urgency += 0.8
        elif intent == 'complaint':
            urgency += 0.6
        elif intent == 'request':
            urgency += 0.3
        
        # Sentiment-based urgency (negative sentiment increases urgency)
        if sentiment < -0.5:
            urgency += 0.4
        
        # Text pattern urgency
        urgency_patterns = [
            r'\b(urgent|emergency|asap|immediately|critical)\b',
            r'!!+',
            r'\b(quick|fast|hurry|rush)\b'
        ]
        
        for pattern in urgency_patterns:
            if re.search(pattern, text.lower()):
                urgency += 0.2
        
        return min(1.0, urgency)
    
    def _identify_topics(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Basic topic identification"""
        topics = []
        text_lower = text.lower()
        
        # Technology topics
        tech_keywords = ['ai', 'artificial intelligence', 'machine learning', 'python', 
                        'programming', 'code', 'software', 'computer', 'technology']
        if any(keyword in text_lower for keyword in tech_keywords):
            topics.append('technology')
        
        # Business topics
        business_keywords = ['business', 'company', 'meeting', 'project', 'work', 
                           'client', 'customer', 'sales', 'marketing']
        if any(keyword in text_lower for keyword in business_keywords):
            topics.append('business')
        
        # Personal topics
        personal_keywords = ['family', 'friend', 'personal', 'hobby', 'vacation', 
                           'health', 'home', 'life']
        if any(keyword in text_lower for keyword in personal_keywords):
            topics.append('personal')
        
        # Add entity-based topics
        for entity in entities:
            if entity['type'] == 'email':
                topics.append('communication')
            elif entity['type'] == 'money':
                topics.append('financial')
            elif entity['type'] == 'date':
                topics.append('scheduling')
        
        return list(set(topics))  # Remove duplicates
    
    async def cleanup(self) -> bool:
        """Clean up resources"""
        try:
            self.is_initialized = False
            logger.info("Perception processor cleaned up")
            return True
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False


# src/agents/perception/__init__.py
"""
Enhanced Perception Module

Provides comprehensive input analysis including:
- Text preprocessing and cleaning
- Intent classification  
- Entity extraction
- Sentiment analysis
- Complexity and urgency scoring
- Topic identification
- Optional LLM enhancement
"""

from .processor import PerceptionProcessor, PerceptionResult

__all__ = ["PerceptionProcessor", "PerceptionResult"]