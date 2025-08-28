# src/integrations/databases/vector_store.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import logging
import chromadb
from chromadb.config import Settings
from config.settings import settings
import json
import hashlib

logger = logging.getLogger(__name__)

class VectorStore(ABC):
   """Abstract base class for vector databases"""
   
   @abstractmethod
   async def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], 
                          ids: List[str] = None) -> bool:
       """Add documents to the vector store"""
       pass
   
   @abstractmethod
   async def similarity_search(self, query: str, k: int = 5, 
                              metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
       """Search for similar documents"""
       pass
   
   @abstractmethod
   async def delete_documents(self, ids: List[str]) -> bool:
       """Delete documents by IDs"""
       pass
   
   @abstractmethod
   def get_collection_stats(self) -> Dict[str, Any]:
       """Get statistics about the collection"""
       pass

class ChromaVectorStore(VectorStore):
   """ChromaDB implementation with enhanced features for agent memory"""
   
   def __init__(self, collection_name: str = "agent_memories"):
       self.collection_name = collection_name
       
       # Initialize ChromaDB client
       self.client = chromadb.PersistentClient(
           path=settings.chroma_persist_directory,
           settings=Settings(
               anonymized_telemetry=False,
               allow_reset=True
           )
       )
       
       # Create or get collection
       self.collection = self._get_or_create_collection()
       
       # Memory management settings
       self.max_memories = getattr(settings, 'max_memories', 10000)
       
       logger.info(f"Initialized ChromaDB vector store: {collection_name}")
   
   def _get_or_create_collection(self):
       """Get or create ChromaDB collection with metadata"""
       try:
           # Try to get existing collection
           return self.client.get_collection(name=self.collection_name)
       except Exception:
           # Create new collection if it doesn't exist
           return self.client.create_collection(
               name=self.collection_name,
               metadata={
                   "description": "AI Agent memory storage with enhanced retrieval",
                   "created_at": datetime.now().isoformat(),
                   "version": "1.0"
               }
           )
   
   async def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], 
                          ids: List[str] = None) -> bool:
       """Add documents to ChromaDB with enhanced metadata"""
       try:
           if not documents:
               return False
           
           # Generate IDs if not provided
           if ids is None:
               ids = [
                   hashlib.md5(f"{doc}_{datetime.now().isoformat()}_{i}".encode()).hexdigest()
                   for i, doc in enumerate(documents)
               ]
           
           # Enhance metadata with timestamps and indexing info
           enhanced_metadatas = []
           for i, metadata in enumerate(metadatas):
               enhanced_metadata = {
                   'timestamp': datetime.now().isoformat(),
                   'content_length': len(documents[i]),
                   'word_count': len(documents[i].split()),
                   'document_index': i,
                   **metadata
               }
               enhanced_metadatas.append(enhanced_metadata)
           
           # Add to collection
           self.collection.add(
               documents=documents,
               metadatas=enhanced_metadatas,
               ids=ids
           )
           
           logger.info(f"Added {len(documents)} documents to ChromaDB collection")
           
           # Check if we need to manage memory capacity
           await self._manage_memory_capacity()
           
           return True
           
       except Exception as e:
           logger.error(f"Failed to add documents to ChromaDB: {e}")
           return False
   
   async def similarity_search(self, query: str, k: int = 5, 
                              metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
       """Search for similar documents with enhanced scoring"""
       try:
           # Build where clause for filtering
           where_clause = metadata_filter if metadata_filter else None
           
           # Perform the search
           results = self.collection.query(
               query_texts=[query],
               n_results=k,
               where=where_clause,
               include=['documents', 'metadatas', 'distances']
           )
           
           if not results['ids'][0]:
               return []
           
           # Format results with enhanced information
           formatted_results = []
           for i in range(len(results['ids'][0])):
               result = {
                   'id': results['ids'][0][i],
                   'document': results['documents'][0][i],
                   'metadata': results['metadatas'][0][i],
                   'distance': results['distances'][0][i],
                   'similarity_score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                   'relevance_score': self._calculate_relevance_score(
                       results['distances'][0][i], 
                       results['metadatas'][0][i]
                   )
               }
               formatted_results.append(result)
           
           # Sort by relevance score (highest first)
           formatted_results.sort(key=lambda x: x['relevance_score'], reverse=True)
           
           return formatted_results
           
       except Exception as e:
           logger.error(f"Failed to search ChromaDB: {e}")
           return []
   
   def _calculate_relevance_score(self, distance: float, metadata: Dict[str, Any]) -> float:
       """Calculate enhanced relevance score considering recency and importance"""
       base_similarity = 1.0 - distance
       
       # Time decay factor (more recent = higher score)
       time_factor = 1.0
       timestamp_str = metadata.get('timestamp')
       if timestamp_str:
           try:
               timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
               age_hours = (datetime.now() - timestamp).total_seconds() / 3600
               # Exponential decay: recent memories are weighted higher
               time_factor = np.exp(-age_hours / 168)  # Decay over a week
           except:
               pass
       
       # Importance factor
       importance = metadata.get('importance', 0.5)
       
       # Content length factor (slightly favor longer, more detailed content)
       content_length = metadata.get('content_length', 0)
       length_factor = min(1.2, 1.0 + (content_length / 1000) * 0.2)
       
       # Combine factors
       relevance_score = base_similarity * (0.6 + time_factor * 0.2 + importance * 0.2) * length_factor
       
       return min(1.0, relevance_score)
   
   async def delete_documents(self, ids: List[str]) -> bool:
       """Delete documents from ChromaDB"""
       try:
           if not ids:
               return True
           
           self.collection.delete(ids=ids)
           logger.info(f"Deleted {len(ids)} documents from ChromaDB")
           return True
           
       except Exception as e:
           logger.error(f"Failed to delete documents from ChromaDB: {e}")
           return False
   
   async def _manage_memory_capacity(self):
       """Manage memory capacity by removing oldest, least important memories"""
       try:
           current_count = self.collection.count()
           
           if current_count <= self.max_memories:
               return
           
           logger.info(f"Memory capacity exceeded ({current_count}/{self.max_memories}), cleaning up...")
           
           # Get all documents with metadata to determine which to remove
           all_docs = self.collection.get(
               include=['metadatas'],
               limit=current_count
           )
           
           # Score documents for removal (lower score = more likely to be removed)
           removal_candidates = []
           for i, metadata in enumerate(all_docs['metadatas']):
               removal_score = self._calculate_removal_score(metadata)
               removal_candidates.append({
                   'id': all_docs['ids'][i],
                   'score': removal_score,
                   'metadata': metadata
               })
           
           # Sort by removal score (ascending - remove lowest scores first)
           removal_candidates.sort(key=lambda x: x['score'])
           
           # Calculate how many to remove
           excess_count = current_count - self.max_memories
           ids_to_remove = [candidate['id'] for candidate in removal_candidates[:excess_count]]
           
           # Remove excess memories
           if ids_to_remove:
               await self.delete_documents(ids_to_remove)
               logger.info(f"Removed {len(ids_to_remove)} old memories to maintain capacity")
               
       except Exception as e:
           logger.error(f"Memory capacity management failed: {e}")
   
   def _calculate_removal_score(self, metadata: Dict[str, Any]) -> float:
       """Calculate score for memory removal (lower = more likely to be removed)"""
       base_score = 0.5
       
       # Age factor (older = lower score)
       timestamp_str = metadata.get('timestamp')
       if timestamp_str:
           try:
               timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
               age_days = (datetime.now() - timestamp).total_seconds() / 86400
               age_factor = max(0.1, 1.0 - (age_days / 30))  # Decay over 30 days
               base_score *= age_factor
           except:
               base_score *= 0.5  # Penalize unparseable timestamps
       
       # Importance factor
       importance = metadata.get('importance', 0.5)
       base_score += importance * 0.3
       
       # Usage factor (if we track access counts)
       access_count = metadata.get('access_count', 0)
       usage_factor = min(0.2, access_count / 10 * 0.2)  # Cap at 0.2
       base_score += usage_factor
       
       return base_score
   
   def get_collection_stats(self) -> Dict[str, Any]:
       """Get comprehensive statistics about the collection"""
       try:
           count = self.collection.count()
           
           if count == 0:
               return {
                   'collection_name': self.collection_name,
                   'document_count': 0,
                   'status': 'empty'
               }
           
           # Get sample of documents for analysis
           sample_size = min(count, 100)
           sample_docs = self.collection.get(
               include=['metadatas'],
               limit=sample_size
           )
           
           # Analyze metadata
           content_lengths = []
           word_counts = []
           timestamps = []
           importance_scores = []
           
           for metadata in sample_docs['metadatas']:
               content_lengths.append(metadata.get('content_length', 0))
               word_counts.append(metadata.get('word_count', 0))
               importance_scores.append(metadata.get('importance', 0.5))
               
               timestamp_str = metadata.get('timestamp')
               if timestamp_str:
                   try:
                       timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                       timestamps.append(timestamp)
                   except:
                       pass
           
           stats = {
               'collection_name': self.collection_name,
               'document_count': count,
               'capacity_utilization': (count / self.max_memories) * 100,
               'sample_size': sample_size,
               'content_stats': {
                   'avg_length': np.mean(content_lengths) if content_lengths else 0,
                   'avg_word_count': np.mean(word_counts) if word_counts else 0,
                   'avg_importance': np.mean(importance_scores) if importance_scores else 0.5
               },
               'time_range': {
                   'oldest': min(timestamps).isoformat() if timestamps else None,
                   'newest': max(timestamps).isoformat() if timestamps else None,
                   'span_days': (max(timestamps) - min(timestamps)).days if len(timestamps) > 1 else 0
               },
               'status': 'active',
               'last_analyzed': datetime.now().isoformat()
           }
           
           return stats
           
       except Exception as e:
           logger.error(f"Failed to get collection stats: {e}")
           return {
               'collection_name': self.collection_name,
               'error': str(e),
               'status': 'error'
           }
   
   async def search_by_metadata(self, metadata_filter: Dict[str, Any], 
                               limit: int = 10) -> List[Dict[str, Any]]:
       """Search documents by metadata criteria only"""
       try:
           results = self.collection.get(
               where=metadata_filter,
               limit=limit,
               include=['documents', 'metadatas']
           )
           
           formatted_results = []
           for i in range(len(results['ids'])):
               formatted_results.append({
                   'id': results['ids'][i],
                   'document': results['documents'][i],
                   'metadata': results['metadatas'][i]
               })
           
           return formatted_results
           
       except Exception as e:
           logger.error(f"Metadata search failed: {e}")
           return []
   
   async def update_document_metadata(self, doc_id: str, 
                                    new_metadata: Dict[str, Any]) -> bool:
       """Update metadata for a specific document"""
       try:
           # Get current document
           current_doc = self.collection.get(
               ids=[doc_id],
               include=['metadatas']
           )
           
           if not current_doc['metadatas']:
               return False
           
           # Merge with existing metadata
           updated_metadata = {**current_doc['metadatas'][0], **new_metadata}
           updated_metadata['last_updated'] = datetime.now().isoformat()
           
           # Update the document
           self.collection.update(
               ids=[doc_id],
               metadatas=[updated_metadata]
           )
           
           return True
           
       except Exception as e:
           logger.error(f"Failed to update document metadata: {e}")
           return False

# Alternative implementations for other vector databases
class PineconeVectorStore(VectorStore):
   """Pinecone implementation (placeholder - requires Pinecone setup)"""
   
   def __init__(self, collection_name: str = "agent_memories"):
       # This would require pinecone-client setup
       raise NotImplementedError("Pinecone integration requires additional setup")

class FAISSVectorStore(VectorStore):
   """FAISS implementation for local vector search"""
   
   def __init__(self, collection_name: str = "agent_memories"):
       # This would use FAISS for local vector operations
       raise NotImplementedError("FAISS integration not yet implemented")

# Factory function to create vector store based on configuration
def create_vector_store(collection_name: str = "agent_memories") -> VectorStore:
   """Create vector store based on configuration"""
   vector_db_type = settings.vector_db_type.lower()
   
   if vector_db_type == "chroma":
       return ChromaVectorStore(collection_name)
   elif vector_db_type == "pinecone":
       return PineconeVectorStore(collection_name)
   elif vector_db_type == "faiss":
       return FAISSVectorStore(collection_name)
   else:
       logger.warning(f"Unknown vector DB type: {vector_db_type}, defaulting to Chroma")
       return ChromaVectorStore(collection_name)

# Integration with Chapter 2 memory components
class VectorMemoryIntegration:
   """Integration layer between Chapter 2 memory components and vector database"""
   
   def __init__(self, vector_store: VectorStore):
       self.vector_store = vector_store
       self.conversation_buffer = []
       self.max_buffer_size = 10
   
   async def store_agent_memory(self, observation: Dict[str, Any], 
                               action: Dict[str, Any], result: Dict[str, Any]) -> bool:
       """Store agent interaction as searchable memory"""
       try:
           # Create a narrative description of the interaction
           memory_content = self._create_memory_narrative(observation, action, result)
           
           # Extract metadata
           metadata = {
               'type': 'agent_interaction',
               'intent': observation.get('intent', 'unknown'),
               'action_type': action.get('action_type', 'unknown'),
               'success': result.get('success', False),
               'importance': self._calculate_interaction_importance(observation, action, result),
               'user_id': observation.get('user_id', 'default'),
               'processing_time': result.get('execution_time', 0)
           }
           
           # Store in vector database
           success = await self.vector_store.add_documents(
               documents=[memory_content],
               metadatas=[metadata]
           )
           
           return success
           
       except Exception as e:
           logger.error(f"Failed to store agent memory: {e}")
           return False
   
   def _create_memory_narrative(self, observation: Dict[str, Any], 
                              action: Dict[str, Any], result: Dict[str, Any]) -> str:
       """Create a natural language narrative from agent interaction"""
       user_input = observation.get('raw_text', 'Unknown input')
       intent = observation.get('intent', 'unknown intent')
       action_type = action.get('action_type', 'took action')
       success = result.get('success', False)
       
       narrative = f"User said: '{user_input}' "
       narrative += f"(Intent: {intent}). "
       narrative += f"Agent {action_type}. "
       narrative += f"Result: {'Success' if success else 'Failed'}."
       
       # Add any specific result details
       if 'content' in result:
           narrative += f" Output: {result['content'][:200]}..."
       
       return narrative
   
   def _calculate_interaction_importance(self, observation: Dict[str, Any], 
                                       action: Dict[str, Any], result: Dict[str, Any]) -> float:
       """Calculate importance score for the interaction"""
       base_importance = 0.5
       
       # Boost importance for successful interactions
       if result.get('success', False):
           base_importance += 0.2
       
       # Boost for certain intents
       intent = observation.get('intent', '')
       if intent in ['complaint', 'urgent', 'important']:
           base_importance += 0.3
       
       # Boost for complex actions
       action_type = action.get('action_type', '')
       complex_actions = ['search_information', 'escalate_issue', 'call_api']
       if action_type in complex_actions:
           base_importance += 0.2
       
       # Consider user sentiment
       sentiment = observation.get('sentiment', 0)
       if abs(sentiment) > 0.7:  # Strong positive or negative sentiment
           base_importance += 0.1
       
       return min(1.0, base_importance)
   
   async def retrieve_relevant_memories(self, query: str, context: Dict[str, Any] = None, 
                                      max_results: int = 5) -> List[Dict[str, Any]]:
       """Retrieve memories relevant to current query/context"""
       try:
           # Build metadata filter if context provided
           metadata_filter = {}
           if context:
               if 'user_id' in context:
                   metadata_filter['user_id'] = context['user_id']
               if 'intent' in context:
                   metadata_filter['intent'] = context['intent']
           
           # Search vector database
           results = await self.vector_store.similarity_search(
               query=query,
               k=max_results,
               metadata_filter=metadata_filter if metadata_filter else None
           )
           
           # Format for agent consumption
           formatted_memories = []
           for result in results:
               if result['similarity_score'] > 0.3:  # Filter out very low similarity
                   formatted_memories.append({
                       'content': result['document'],
                       'relevance': result['similarity_score'],
                       'metadata': result['metadata'],
                       'memory_type': result['metadata'].get('type', 'unknown')
                   })
           
           return formatted_memories
           
       except Exception as e:
           logger.error(f"Memory retrieval failed: {e}")
           return []