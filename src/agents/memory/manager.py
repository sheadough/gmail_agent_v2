# src/agents/memory/manager.py
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path

from ..base.component import BaseComponent

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memory storage"""
    WORKING = "working"      # Short-term, current session
    EPISODIC = "episodic"    # Specific experiences/interactions
    SEMANTIC = "semantic"    # Facts and knowledge
    PROCEDURAL = "procedural" # How to do things
    CONTEXTUAL = "contextual" # User preferences and context

@dataclass
class MemoryItem:
    """Individual memory item"""
    id: str
    content: Union[str, Dict[str, Any]]
    memory_type: MemoryType
    importance: float = 0.5
    confidence: float = 1.0
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class WorkingMemory:
    """Short-term working memory for current session"""
    
    def __init__(self, max_items: int = 50):
        self.max_items = max_items
        self.items: List[MemoryItem] = []
        self.current_context = {}
    
    def add(self, content: Any, importance: float = 0.5, tags: List[str] = None):
        """Add item to working memory"""
        item = MemoryItem(
            id=f"wm_{datetime.now().timestamp()}",
            content=content,
            memory_type=MemoryType.WORKING,
            importance=importance,
            tags=tags or []
        )
        
        self.items.append(item)
        
        # Keep only most recent/important items
        if len(self.items) > self.max_items:
            # Sort by importance and recency
            self.items.sort(key=lambda x: (x.importance, x.created_at.timestamp()), reverse=True)
            self.items = self.items[:self.max_items]
    
    def get_recent(self, count: int = 10) -> List[MemoryItem]:
        """Get recent items from working memory"""
        sorted_items = sorted(self.items, key=lambda x: x.created_at, reverse=True)
        return sorted_items[:count]
    
    def search(self, query: str, max_results: int = 5) -> List[MemoryItem]:
        """Simple text search in working memory"""
        query_lower = query.lower()
        results = []
        
        for item in self.items:
            content_str = str(item.content).lower()
            if query_lower in content_str or any(query_lower in tag.lower() for tag in item.tags):
                results.append(item)
        
        return results[:max_results]
    
    def clear(self):
        """Clear working memory"""
        self.items.clear()
        self.current_context.clear()

class EpisodicMemory:
    """Long-term episodic memory for experiences"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path) if storage_path else Path("data/episodic_memory.json")
        self.episodes: List[MemoryItem] = []
        self.max_episodes = 1000
        
    async def initialize(self):
        """Load existing episodes"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                for episode_data in data.get('episodes', []):
                    episode = MemoryItem(
                        id=episode_data['id'],
                        content=episode_data['content'],
                        memory_type=MemoryType(episode_data['memory_type']),
                        importance=episode_data['importance'],
                        confidence=episode_data['confidence'],
                        access_count=episode_data['access_count'],
                        created_at=datetime.fromisoformat(episode_data['created_at']),
                        last_accessed=datetime.fromisoformat(episode_data['last_accessed']),
                        tags=episode_data.get('tags', []),
                        metadata=episode_data.get('metadata', {})
                    )
                    
                    if episode_data.get('expires_at'):
                        episode.expires_at = datetime.fromisoformat(episode_data['expires_at'])
                    
                    self.episodes.append(episode)
                    
                logger.info(f"Loaded {len(self.episodes)} episodes from storage")
                
        except Exception as e:
            logger.error(f"Failed to load episodic memory: {e}")
    
    def add_episode(self, interaction: Dict[str, Any], importance: float = 0.5):
        """Add new episode to memory"""
        episode = MemoryItem(
            id=f"ep_{datetime.now().timestamp()}",
            content=interaction,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            tags=interaction.get('tags', [])
        )
        
        self.episodes.append(episode)
        
        # Manage memory capacity
        if len(self.episodes) > self.max_episodes:
            self._cleanup_old_episodes()
    
    def _cleanup_old_episodes(self):
        """Remove old/unimportant episodes"""
        # Sort by importance and recency
        self.episodes.sort(
            key=lambda x: (x.importance, x.last_accessed.timestamp()),
            reverse=True
        )
        
        # Remove least important episodes
        removed_count = len(self.episodes) - self.max_episodes
        if removed_count > 0:
            self.episodes = self.episodes[:-removed_count]
            logger.info(f"Removed {removed_count} old episodes from memory")
    
    def get_related_episodes(self, query: str, context: Dict[str, Any] = None, max_results: int = 5) -> List[MemoryItem]:
        """Find episodes related to query/context"""
        query_lower = query.lower()
        scored_episodes = []
        
        for episode in self.episodes:
            score = self._calculate_relevance_score(episode, query_lower, context)
            if score > 0.1:  # Minimum relevance threshold
                scored_episodes.append((episode, score))
        
        # Sort by relevance score
        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        
        # Update access counts and return episodes
        results = []
        for episode, score in scored_episodes[:max_results]:
            episode.access_count += 1
            episode.last_accessed = datetime.now()
            results.append(episode)
        
        return results
    
    def _calculate_relevance_score(self, episode: MemoryItem, query: str, context: Dict[str, Any] = None) -> float:
        """Calculate relevance score for an episode"""
        score = 0.0
        content_str = str(episode.content).lower()
        
        # Text similarity (simple keyword matching)
        if query in content_str:
            score += 0.5
        
        # Tag matching
        for tag in episode.tags:
            if query in tag.lower():
                score += 0.3
        
        # Context matching
        if context:
            user_id = context.get('user_id')
            if user_id and isinstance(episode.content, dict):
                if episode.content.get('user_id') == user_id:
                    score += 0.2
        
        # Recency bonus (recent episodes slightly preferred)
        days_old = (datetime.now() - episode.created_at).days
        recency_factor = max(0, 1 - days_old / 30)  # Decay over 30 days
        score += recency_factor * 0.1
        
        # Importance factor
        score += episode.importance * 0.2
        
        return min(1.0, score)
    
    async def save(self):
        """Save episodes to storage"""
        try:
            # Prepare data for serialization
            episodes_data = []
            for episode in self.episodes:
                episode_data = {
                    'id': episode.id,
                    'content': episode.content,
                    'memory_type': episode.memory_type.value,
                    'importance': episode.importance,
                    'confidence': episode.confidence,
                    'access_count': episode.access_count,
                    'created_at': episode.created_at.isoformat(),
                    'last_accessed': episode.last_accessed.isoformat(),
                    'tags': episode.tags,
                    'metadata': episode.metadata
                }
                
                if episode.expires_at:
                    episode_data['expires_at'] = episode.expires_at.isoformat()
                
                episodes_data.append(episode_data)
            
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.storage_path, 'w') as f:
                json.dump({'episodes': episodes_data}, f, indent=2)
            
            logger.info(f"Saved {len(self.episodes)} episodes to storage")
            
        except Exception as e:
            logger.error(f"Failed to save episodic memory: {e}")

class SemanticMemory:
    """Long-term semantic memory for facts and knowledge"""
    
    def __init__(self):
        self.facts: Dict[str, MemoryItem] = {}
        self.categories = {
            'user_preferences': {},
            'learned_facts': {},
            'procedures': {},
            'relationships': {}
        }
    
    def store_fact(self, key: str, fact: Any, category: str = 'learned_facts', 
                  confidence: float = 1.0, source: str = None):
        """Store a semantic fact"""
        fact_item = MemoryItem(
            id=f"fact_{key}",
            content=fact,
            memory_type=MemoryType.SEMANTIC,
            confidence=confidence,
            tags=[category],
            metadata={'source': source} if source else {}
        )
        
        self.facts[key] = fact_item
        
        if category in self.categories:
            self.categories[category][key] = fact
    
    def get_fact(self, key: str) -> Optional[MemoryItem]:
        """Retrieve a semantic fact"""
        if key in self.facts:
            fact = self.facts[key]
            fact.access_count += 1
            fact.last_accessed = datetime.now()
            return fact
        return None
    
    def search_facts(self, query: str, category: str = None) -> List[MemoryItem]:
        """Search semantic facts"""
        query_lower = query.lower()
        results = []
        
        for key, fact in self.facts.items():
            # Check if matches category filter
            if category and category not in fact.tags:
                continue
            
            # Check text match
            if (query_lower in key.lower() or 
                query_lower in str(fact.content).lower()):
                results.append(fact)
        
        return results

class MemoryConsolidationSystem:
    """System for consolidating memories between different types"""
    
    def __init__(self, episodic_memory: EpisodicMemory, semantic_memory: SemanticMemory):
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
    
    def consolidate_working_to_long_term(self, working_memory: WorkingMemory):
        """Move important items from working memory to long-term storage"""
        for item in working_memory.items:
            if item.importance > 0.7:  # High importance threshold
                # Convert to episodic memory
                episode_data = {
                    'content': item.content,
                    'importance': item.importance,
                    'created_at': item.created_at.isoformat(),
                    'tags': item.tags,
                    'metadata': item.metadata
                }
                
                self.episodic_memory.add_episode(episode_data, item.importance)
        
        # Clear working memory after consolidation
        working_memory.clear()
    
    def extract_semantic_knowledge(self):
        """Extract semantic knowledge from episodic memories"""
        # Look for patterns in episodes that should become semantic facts
        user_preferences = {}
        
        for episode in self.episodic_memory.episodes:
            if isinstance(episode.content, dict):
                # Extract user preferences
                if 'user_preference' in episode.content:
                    pref_key = episode.content.get('preference_type')
                    pref_value = episode.content.get('preference_value')
                    
                    if pref_key and pref_value:
                        user_preferences[pref_key] = pref_value
        
        # Store as semantic facts
        for key, value in user_preferences.items():
            self.semantic_memory.store_fact(
                key, value, 'user_preferences', 
                confidence=0.8, source='episodic_consolidation'
            )

class MemoryManager(BaseComponent):
    """Main memory management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("memory_manager", config)
        
        # Initialize memory systems
        self.working_memory = WorkingMemory(
            max_items=config.get('working_memory_size', 50)
        )
        
        storage_path = config.get('episodic_storage_path', 'data/episodic_memory.json')
        self.episodic_memory = EpisodicMemory(storage_path)
        
        self.semantic_memory = SemanticMemory()
        
        self.consolidation_system = MemoryConsolidationSystem(
            self.episodic_memory, self.semantic_memory
        )
        
        # Vector store integration (if available)
        self.vector_store = None
        if config.get('use_vector_store'):
            try:
                from ...integrations.databases.vector_store import create_vector_store
                collection_name = config.get('vector_collection', 'agent_memory')
                self.vector_store = create_vector_store(collection_name)
                logger.info("Vector store integration enabled")
            except ImportError:
                logger.warning("Vector store not available")
    
    async def initialize(self) -> bool:
        """Initialize memory systems"""
        try:
            await self.episodic_memory.initialize()
            self.is_initialized = True
            logger.info("Memory manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            return False
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process memory operations"""
        operation = context.get('operation') if context else 'store'
        
        if operation == 'store':
            return await self.store_memory(input_data, context)
        elif operation == 'retrieve':
            return await self.retrieve_memory(input_data, context)
        elif operation == 'search':
            return await self.search_memory(input_data, context)
        else:
            raise ValueError(f"Unknown memory operation: {operation}")
    
    async def store_memory(self, content: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store content in appropriate memory system"""
        memory_type = context.get('memory_type', 'working') if context else 'working'
        importance = context.get('importance', 0.5) if context else 0.5
        tags = context.get('tags', []) if context else []
        
        if memory_type == 'working':
            self.working_memory.add(content, importance, tags)
            
        elif memory_type == 'episodic':
            episode_data = {
                'content': content,
                'user_id': context.get('user_id') if context else None,
                'timestamp': datetime.now().isoformat(),
                'tags': tags,
                'metadata': context.get('metadata', {}) if context else {}
            }
            self.episodic_memory.add_episode(episode_data, importance)
            
        elif memory_type == 'semantic':
            key = context.get('key') if context else f"fact_{datetime.now().timestamp()}"
            category = context.get('category', 'learned_facts') if context else 'learned_facts'
            confidence = context.get('confidence', 1.0) if context else 1.0
            
            self.semantic_memory.store_fact(key, content, category, confidence)
        
        # Also store in vector database if available
        if self.vector_store:
            try:
                metadata = {
                    'memory_type': memory_type,
                    'importance': importance,
                    'timestamp': datetime.now().isoformat(),
                    **context.get('metadata', {}) if context else {}
                }
                
                await self.vector_store.add_documents([str(content)], [metadata])
                
            except Exception as e:
                logger.warning(f"Failed to store in vector database: {e}")
        
        return {'status': 'stored', 'memory_type': memory_type}
    
    async def retrieve_memory(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retrieve relevant memories"""
        results = {
            'working_memory': [],
            'episodic_memory': [],
            'semantic_memory': [],
            'vector_memory': []
        }
        
        # Search working memory
        results['working_memory'] = self.working_memory.search(query)
        
        # Search episodic memory
        results['episodic_memory'] = self.episodic_memory.get_related_episodes(
            query, context, max_results=5
        )
        
        # Search semantic memory
        category = context.get('category') if context else None
        results['semantic_memory'] = self.semantic_memory.search_facts(query, category)
        
        # Search vector memory if available
        if self.vector_store:
            try:
                vector_results = await self.vector_store.similarity_search(
                    query, k=5, 
                    metadata_filter=context.get('metadata_filter') if context else None
                )
                results['vector_memory'] = vector_results
            except Exception as e:
                logger.warning(f"Vector memory search failed: {e}")
        
        return results
    
    async def search_memory(self, query: str, context: Dict[str, Any] = None) -> List[MemoryItem]:
        """Unified memory search across all systems"""
        all_results = []
        
        # Get results from all memory systems
        memory_results = await self.retrieve_memory(query, context)
        
        # Combine and deduplicate results
        for memory_type, items in memory_results.items():
            if memory_type == 'vector_memory':
                # Convert vector results to MemoryItem format
                for result in items:
                    if isinstance(result, dict) and 'document' in result:
                        item = MemoryItem(
                            id=result.get('id', f"vector_{datetime.now().timestamp()}"),
                            content=result['document'],
                            memory_type=MemoryType.SEMANTIC,
                            confidence=result.get('similarity_score', 0.5),
                            metadata=result.get('metadata', {})
                        )
                        all_results.append(item)
            else:
                all_results.extend(items)
        
        # Sort by relevance/importance
        all_results.sort(key=lambda x: (x.importance, x.confidence), reverse=True)
        
        return all_results[:10]  # Return top 10 results
    
    async def consolidate_memories(self):
        """Perform memory consolidation"""
        try:
            # Consolidate working memory to long-term storage
            self.consolidation_system.consolidate_working_to_long_term(self.working_memory)
            
            # Extract semantic knowledge from episodes
            self.consolidation_system.extract_semantic_knowledge()
            
            # Save episodic memories
            await self.episodic_memory.save()
            
            logger.info("Memory consolidation completed successfully")
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
    
    async def cleanup(self) -> bool:
        """Clean up memory systems"""
        try:
            # Save all memories before cleanup
            await self.episodic_memory.save()
            
            # Clear working memory
            self.working_memory.clear()
            
            self.is_initialized = False
            logger.info("Memory manager cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        return {
            'working_memory': {
                'item_count': len(self.working_memory.items),
                'max_items': self.working_memory.max_items
            },
            'episodic_memory': {
                'episode_count': len(self.episodic_memory.episodes),
                'max_episodes': self.episodic_memory.max_episodes
            },
            'semantic_memory': {
                'fact_count': len(self.semantic_memory.facts),
                'categories': list(self.semantic_memory.categories.keys())
            },
            'vector_store': {
                'enabled': self.vector_store is not None,
                'stats': self.vector_store.get_collection_stats() if self.vector_store else {}
            }
        }


# src/agents/memory/__init__.py
"""
Advanced Memory Module

Provides multi-layered memory management:
- Working Memory: Short-term session memory
- Episodic Memory: Long-term experience storage
- Semantic Memory: Facts and knowledge storage
- Memory Consolidation: Moving between memory types
- Vector Integration: Enhanced search capabilities
"""

from .manager import (
    MemoryManager, 
    MemoryItem, 
    MemoryType,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory
)

__all__ = [
    "MemoryManager",
    "MemoryItem", 
    "MemoryType",
    "WorkingMemory",
    "EpisodicMemory", 
    "SemanticMemory"
]