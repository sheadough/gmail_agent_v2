# src/integrations/openai/enhanced_perception.py
from typing import Dict, List, Any
import json
import asyncio
from src.integrations.openai.client import openai_client
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LLMEnhancedPerceptionModule:
    """Perception module enhanced with LLM capabilities"""
    
    def __init__(self):
        self.intent_classification_prompt = """
You are an expert at analyzing user messages and determining their intent. 
Classify the following message into one of these categories:
- question: User is asking for information
- request: User wants something done
- complaint: User is expressing dissatisfaction
- compliment: User is expressing appreciation
- greeting: User is saying hello or goodbye
- support: User needs help or assistance
- information: User is providing information
- other: Doesn't fit other categories

Message: "{text}"

Respond with just the category name and a confidence score (0-1).
Format: category_name,confidence_score
"""
        
        self.entity_extraction_prompt = """
Extract important entities from this message. Return as JSON.
Include: names, dates, times, locations, products, amounts, emails, phone numbers, etc.

Message: "{text}"

Return format:
{{
    "entities": [
        {{"type": "person_name", "value": "John Smith", "confidence": 0.95}},
        {{"type": "date", "value": "2024-01-15", "confidence": 0.9}}
    ]
}}
"""
        
        self.sentiment_analysis_prompt = """
Analyze the sentiment of this message on a scale from -1 (very negative) to 1 (very positive).
Also provide the emotional tone and intensity.

Message: "{text}"

Return as JSON:
{{
    "sentiment_score": 0.5,
    "emotional_tone": "frustrated",
    "intensity": "medium",
    "confidence": 0.85
}}
"""
    
    async def enhanced_perceive(self, raw_input: str) -> Dict[str, Any]:
        """Enhanced perception using LLM capabilities"""
        try:
            # Run multiple analysis tasks concurrently
            tasks = [
                self._classify_intent(raw_input),
                self._extract_entities(raw_input),
                self._analyze_sentiment(raw_input)
            ]
            
            intent_result, entities_result, sentiment_result = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            
            # Build comprehensive observation
            observations = {
                'timestamp': datetime.now(),
                'raw_text': raw_input,
                'text_length': len(raw_input),
                'word_count': len(raw_input.split()),
                'enhanced_analysis': True
            }
            
            # Process intent classification
            if not isinstance(intent_result, Exception):
                observations['intent'] = intent_result['intent']
                observations['intent_confidence'] = intent_result['confidence']
            else:
                logger.error(f"Intent classification failed: {intent_result}")
                observations['intent'] = 'unknown'
                observations['intent_confidence'] = 0.0
            
            # Process entity extraction
            if not isinstance(entities_result, Exception):
                observations['entities'] = entities_result['entities']
                observations['entity_count'] = len(entities_result['entities'])
            else:
                logger.error(f"Entity extraction failed: {entities_result}")
                observations['entities'] = []
                observations['entity_count'] = 0
            
            # Process sentiment analysis
            if not isinstance(sentiment_result, Exception):
                observations.update({
                    'sentiment': sentiment_result['sentiment_score'],
                    'emotional_tone': sentiment_result['emotional_tone'],
                    'emotional_intensity': sentiment_result['intensity'],
                    'sentiment_confidence': sentiment_result['confidence']
                })
            else:
                logger.error(f"Sentiment analysis failed: {sentiment_result}")
                observations.update({
                    'sentiment': 0.0,
                    'emotional_tone': 'neutral',
                    'emotional_intensity': 'low',
                    'sentiment_confidence': 0.0
                })
            
            # Additional analysis
            observations['complexity_score'] = self._calculate_complexity(raw_input)
            observations['topics'] = await self._identify_topics(raw_input)
            
            return observations
            
        except Exception as e:
            logger.error(f"Enhanced perception failed: {e}")
            # Fallback to basic analysis
            return self._basic_fallback_perception(raw_input)
    
    async def _classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify user intent using LLM"""
        messages = [
            {"role": "user", "content": self.intent_classification_prompt.format(text=text)}
        ]
        
        response = await openai_client.generate_completion(
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent classification
            max_tokens=50
        )
        
        try:
            # Parse response: "category_name,confidence_score"
            parts = response['content'].strip().split(',')
            intent = parts[0].strip()
            confidence = float(parts[1].strip()) if len(parts) > 1 else 0.5
            
            return {'intent': intent, 'confidence': confidence}
        except:
            return {'intent': 'unknown', 'confidence': 0.0}
    
    async def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities using LLM"""
        messages = [
            {"role": "user", "content": self.entity_extraction_prompt.format(text=text)}
        ]
        
        response = await openai_client.generate_completion(
            messages=messages,
            temperature=0.1,  # Very low temperature for consistent extraction
            max_tokens=500
        )
        
        try:
            result = json.loads(response['content'])
            return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse entity extraction JSON")
            return {'entities': []}
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using LLM"""
        messages = [
            {"role": "user", "content": self.sentiment_analysis_prompt.format(text=text)}
        ]
        
        response = await openai_client.generate_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=200
        )
        
        try:
            result = json.loads(response['content'])
            return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse sentiment analysis JSON")
            return {
                'sentiment_score': 0.0,
                'emotional_tone': 'neutral',
                'intensity': 'low',
                'confidence': 0.0
            }
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        words = text.split()
        sentences = text.split('.')
        
        # Basic complexity metrics
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (avg_word_length * 0.1) + (avg_sentence_length * 0.05))
        return complexity
    
    async def _identify_topics(self, text: str) -> List[str]:
        """Identify main topics in the text"""
        if len(text) < 50:  # Skip topic identification for short texts
            return []
        
        topic_prompt = f"""
Identify the main topics discussed in this message. Return up to 3 topics as a simple list.

Message: "{text}"

Topics (one per line):
"""
        
        messages = [{"role": "user", "content": topic_prompt}]
        
        try:
            response = await openai_client.generate_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=100
            )
            
            topics = [
                topic.strip().replace('- ', '').replace('* ', '') 
                for topic in response['content'].split('\n') 
                if topic.strip()
            ]
            return topics[:3]  # Limit to 3 topics
            
        except Exception as e:
            logger.warning(f"Topic identification failed: {e}")
            return []
    
    def _basic_fallback_perception(self, text: str) -> Dict[str, Any]:
        """Fallback perception when LLM analysis fails"""
        return {
            'timestamp': datetime.now(),
            'raw_text': text,
            'text_length': len(text),
            'word_count': len(text.split()),
            'intent': 'unknown',
            'intent_confidence': 0.0,
            'entities': [],
            'entity_count': 0,
            'sentiment': 0.0,
            'emotional_tone': 'neutral',
            'emotional_intensity': 'low',
            'sentiment_confidence': 0.0,
            'complexity_score': 0.0,
            'topics': [],
            'enhanced_analysis': False,
            'fallback_used': True
        }