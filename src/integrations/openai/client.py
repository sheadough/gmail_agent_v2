# src/integrations/openai/client.py
import openai
from openai import OpenAI
from typing import Dict, List, Any, Optional, AsyncGenerator
import asyncio
import json
from datetime import datetime
from config.settings import settings
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class EnhancedOpenAIClient:
    """Enhanced OpenAI client with rate limiting, error handling, and caching"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = asyncio.Semaphore(settings.max_concurrent_requests)
        self.request_count = 0
        self.cache = {}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion with enhanced error handling and rate limiting"""
        
        async with self.rate_limiter:
            try:
                # Use configured defaults
                model = model or settings.openai_model
                temperature = temperature or settings.openai_temperature
                max_tokens = max_tokens or settings.openai_max_tokens
                
                # Create cache key
                cache_key = self._create_cache_key(messages, model, temperature, max_tokens)
                
                # Check cache for identical requests
                if cache_key in self.cache and not stream:
                    logger.debug("Returning cached response")
                    return self.cache[cache_key]
                
                # Make API call
                self.request_count += 1
                logger.info(f"Making OpenAI API call #{self.request_count}")
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    **kwargs
                )
                
                if stream:
                    return self._handle_streaming_response(response)
                else:
                    result = self._format_response(response)
                    # Cache the result
                    self.cache[cache_key] = result
                    return result
                    
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit exceeded: {e}")
                await asyncio.sleep(60)  # Wait 1 minute
                raise
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in OpenAI call: {e}")
                raise
    
    def _create_cache_key(self, messages: List[Dict[str, str]], model: str, 
                         temperature: float, max_tokens: int) -> str:
        """Create cache key for request deduplication"""
        key_data = {
            'messages': messages,
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _format_response(self, response) -> Dict[str, Any]:
        """Format OpenAI response into standard format"""
        return {
            'content': response.choices[0].message.content,
            'model': response.model,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            'finish_reason': response.choices[0].finish_reason,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_streaming_response(self, response) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming response from OpenAI"""
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield {
                    'content': chunk.choices[0].delta.content,
                    'type': 'content',
                    'timestamp': datetime.now().isoformat()
                }
            elif chunk.choices[0].finish_reason is not None:
                yield {
                    'type': 'finish',
                    'finish_reason': chunk.choices[0].finish_reason,
                    'timestamp': datetime.now().isoformat()
                }
    
    async def generate_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002") -> Dict[str, Any]:
        """Generate embeddings for texts"""
        async with self.rate_limiter:
            try:
                response = self.client.embeddings.create(
                    model=model,
                    input=texts
                )
                
                return {
                    'embeddings': [data.embedding for data in response.data],
                    'model': response.model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'total_tokens': response.usage.total_tokens
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise
    
    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()
        logger.info("OpenAI response cache cleared")

# Create singleton instance
openai_client = EnhancedOpenAIClient()