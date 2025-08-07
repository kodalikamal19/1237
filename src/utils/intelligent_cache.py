"""
Intelligent Caching System for HackRX API
Provides multi-layer caching for documents, embeddings, and query results
to optimize latency and reduce API costs.
"""

import os
import json
import pickle
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass, asdict
from threading import Lock
import gc

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from lru import LRU
    LRU_AVAILABLE = True
except ImportError:
    LRU_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata"""
    data: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    tags: List[str] = None

class IntelligentCache:
    """
    Multi-layer intelligent caching system with:
    - Memory cache (LRU) for hot data
    - Redis cache for distributed caching
    - File cache for persistent storage
    - Smart eviction policies
    - Cache warming and preloading
    """
    
    def __init__(self, 
                 memory_cache_size: int = 1000,
                 redis_url: Optional[str] = None,
                 file_cache_dir: str = "/tmp/hackrx_cache",
                 default_ttl: int = 3600):  # 1 hour default TTL
        
        self.memory_cache_size = memory_cache_size
        self.default_ttl = default_ttl
        self.file_cache_dir = file_cache_dir
        self._locks = {}
        self._global_lock = Lock()
        
        # Initialize memory cache
        if LRU_AVAILABLE:
            self.memory_cache = LRU(memory_cache_size)
            logger.info(f"Memory cache initialized with size {memory_cache_size}")
        else:
            self.memory_cache = {}
            logger.warning("LRU cache not available, using basic dict")
        
        # Initialize Redis cache
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {str(e)}")
                self.redis_client = None
        
        # Initialize file cache
        try:
            os.makedirs(file_cache_dir, exist_ok=True)
            self.file_cache_enabled = True
            logger.info(f"File cache initialized at {file_cache_dir}")
        except Exception as e:
            logger.warning(f"File cache initialization failed: {str(e)}")
            self.file_cache_enabled = False
        
        # Cache statistics
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'redis_hits': 0,
            'redis_misses': 0,
            'file_hits': 0,
            'file_misses': 0,
            'total_sets': 0,
            'total_gets': 0,
            'evictions': 0
        }
    
    def _get_lock(self, key: str) -> Lock:
        """Get or create a lock for a specific key"""
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = Lock()
            return self._locks[key]
    
    def _generate_key(self, namespace: str, identifier: str) -> str:
        """Generate a cache key with namespace"""
        # Create hash of identifier for consistent key length
        id_hash = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        return f"hackrx:{namespace}:{id_hash}"
    
    def _calculate_size(self, data: Any) -> int:
        """Estimate size of data in bytes"""
        try:
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data).encode('utf-8'))
            else:
                return len(pickle.dumps(data))
        except:
            return 1024  # Default estimate
    
    def get(self, namespace: str, identifier: str) -> Optional[Any]:
        """
        Get data from cache with multi-layer lookup.
        Checks memory -> Redis -> file cache in order.
        """
        cache_key = self._generate_key(namespace, identifier)
        self.stats['total_gets'] += 1
        
        with self._get_lock(cache_key):
            # Try memory cache first
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if self._is_valid(entry):
                    entry.access_count += 1
                    self.stats['memory_hits'] += 1
                    logger.debug(f"Memory cache hit: {namespace}")
                    return entry.data
                else:
                    # Remove expired entry
                    del self.memory_cache[cache_key]
            
            self.stats['memory_misses'] += 1
            
            # Try Redis cache
            if self.redis_client:
                try:
                    cached_data = self.redis_client.get(cache_key)
                    if cached_data:
                        entry = pickle.loads(cached_data)
                        if self._is_valid(entry):
                            # Promote to memory cache
                            self.memory_cache[cache_key] = entry
                            entry.access_count += 1
                            self.stats['redis_hits'] += 1
                            logger.debug(f"Redis cache hit: {namespace}")
                            return entry.data
                        else:
                            # Remove expired entry
                            self.redis_client.delete(cache_key)
                except Exception as e:
                    logger.warning(f"Redis get error: {str(e)}")
            
            self.stats['redis_misses'] += 1
            
            # Try file cache
            if self.file_cache_enabled:
                try:
                    file_path = os.path.join(self.file_cache_dir, f"{cache_key}.pkl")
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            entry = pickle.load(f)
                        
                        if self._is_valid(entry):
                            # Promote to higher caches
                            self.memory_cache[cache_key] = entry
                            if self.redis_client:
                                try:
                                    self.redis_client.setex(
                                        cache_key,
                                        int(entry.ttl or self.default_ttl),
                                        pickle.dumps(entry)
                                    )
                                except Exception as e:
                                    logger.warning(f"Redis promotion error: {str(e)}")
                            
                            entry.access_count += 1
                            self.stats['file_hits'] += 1
                            logger.debug(f"File cache hit: {namespace}")
                            return entry.data
                        else:
                            # Remove expired file
                            os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"File cache get error: {str(e)}")
            
            self.stats['file_misses'] += 1
            return None
    
    def set(self, namespace: str, identifier: str, data: Any, 
            ttl: Optional[int] = None, tags: Optional[List[str]] = None) -> bool:
        """
        Set data in cache with intelligent placement across layers.
        """
        cache_key = self._generate_key(namespace, identifier)
        self.stats['total_sets'] += 1
        
        ttl = ttl or self.default_ttl
        size_bytes = self._calculate_size(data)
        
        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            access_count=1,
            size_bytes=size_bytes,
            ttl=ttl,
            tags=tags or []
        )
        
        with self._get_lock(cache_key):
            try:
                # Always set in memory cache
                self.memory_cache[cache_key] = entry
                
                # Set in Redis if available and data is not too large
                if self.redis_client and size_bytes < 1024 * 1024:  # 1MB limit for Redis
                    try:
                        self.redis_client.setex(
                            cache_key,
                            ttl,
                            pickle.dumps(entry)
                        )
                    except Exception as e:
                        logger.warning(f"Redis set error: {str(e)}")
                
                # Set in file cache for persistence (for important data)
                if (self.file_cache_enabled and 
                    (namespace in ['documents', 'embeddings'] or size_bytes > 1024 * 1024)):
                    try:
                        file_path = os.path.join(self.file_cache_dir, f"{cache_key}.pkl")
                        with open(file_path, 'wb') as f:
                            pickle.dump(entry, f)
                    except Exception as e:
                        logger.warning(f"File cache set error: {str(e)}")
                
                logger.debug(f"Cache set: {namespace}, size: {size_bytes} bytes")
                return True
                
            except Exception as e:
                logger.error(f"Cache set error: {str(e)}")
                return False
    
    def _is_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid"""
        if entry.ttl is None:
            return True
        
        return (time.time() - entry.timestamp) < entry.ttl
    
    def delete(self, namespace: str, identifier: str) -> bool:
        """Delete entry from all cache layers"""
        cache_key = self._generate_key(namespace, identifier)
        
        with self._get_lock(cache_key):
            deleted = False
            
            # Delete from memory cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
                deleted = True
            
            # Delete from Redis
            if self.redis_client:
                try:
                    self.redis_client.delete(cache_key)
                    deleted = True
                except Exception as e:
                    logger.warning(f"Redis delete error: {str(e)}")
            
            # Delete from file cache
            if self.file_cache_enabled:
                try:
                    file_path = os.path.join(self.file_cache_dir, f"{cache_key}.pkl")
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                        deleted = True
                except Exception as e:
                    logger.warning(f"File delete error: {str(e)}")
            
            return deleted
    
    def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace"""
        cleared_count = 0
        namespace_prefix = f"hackrx:{namespace}:"
        
        # Clear from memory cache
        keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(namespace_prefix)]
        for key in keys_to_delete:
            del self.memory_cache[key]
            cleared_count += 1
        
        # Clear from Redis
        if self.redis_client:
            try:
                redis_keys = self.redis_client.keys(f"{namespace_prefix}*")
                if redis_keys:
                    self.redis_client.delete(*redis_keys)
                    cleared_count += len(redis_keys)
            except Exception as e:
                logger.warning(f"Redis clear error: {str(e)}")
        
        # Clear from file cache
        if self.file_cache_enabled:
            try:
                for filename in os.listdir(self.file_cache_dir):
                    if filename.startswith(namespace_prefix) and filename.endswith('.pkl'):
                        os.unlink(os.path.join(self.file_cache_dir, filename))
                        cleared_count += 1
            except Exception as e:
                logger.warning(f"File clear error: {str(e)}")
        
        logger.info(f"Cleared {cleared_count} entries from namespace: {namespace}")
        return cleared_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['total_gets']
        
        if total_requests > 0:
            memory_hit_rate = self.stats['memory_hits'] / total_requests
            redis_hit_rate = self.stats['redis_hits'] / total_requests
            file_hit_rate = self.stats['file_hits'] / total_requests
            overall_hit_rate = (self.stats['memory_hits'] + 
                              self.stats['redis_hits'] + 
                              self.stats['file_hits']) / total_requests
        else:
            memory_hit_rate = redis_hit_rate = file_hit_rate = overall_hit_rate = 0
        
        return {
            'total_requests': total_requests,
            'overall_hit_rate': overall_hit_rate,
            'memory_hit_rate': memory_hit_rate,
            'redis_hit_rate': redis_hit_rate,
            'file_hit_rate': file_hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'redis_available': self.redis_client is not None,
            'file_cache_available': self.file_cache_enabled,
            **self.stats
        }
    
    def warm_cache(self, warm_data: List[Tuple[str, str, Any]]) -> int:
        """Warm cache with predefined data"""
        warmed_count = 0
        
        for namespace, identifier, data in warm_data:
            try:
                if self.set(namespace, identifier, data):
                    warmed_count += 1
            except Exception as e:
                logger.warning(f"Cache warming error: {str(e)}")
        
        logger.info(f"Warmed cache with {warmed_count} entries")
        return warmed_count
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries from all caches"""
        cleaned_count = 0
        current_time = time.time()
        
        # Clean memory cache
        expired_keys = []
        for key, entry in self.memory_cache.items():
            if entry.ttl and (current_time - entry.timestamp) > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            cleaned_count += 1
        
        # Clean file cache
        if self.file_cache_enabled:
            try:
                for filename in os.listdir(self.file_cache_dir):
                    if filename.endswith('.pkl'):
                        file_path = os.path.join(self.file_cache_dir, filename)
                        try:
                            with open(file_path, 'rb') as f:
                                entry = pickle.load(f)
                            
                            if entry.ttl and (current_time - entry.timestamp) > entry.ttl:
                                os.unlink(file_path)
                                cleaned_count += 1
                        except:
                            # If we can't read the file, remove it
                            os.unlink(file_path)
                            cleaned_count += 1
            except Exception as e:
                logger.warning(f"File cleanup error: {str(e)}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired cache entries")
        
        return cleaned_count

class DocumentCache:
    """Specialized cache for document processing"""
    
    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.namespace = "documents"
    
    def get_document_text(self, url: str) -> Optional[str]:
        """Get cached document text by URL"""
        return self.cache.get(self.namespace, f"text:{url}")
    
    def set_document_text(self, url: str, text: str, ttl: int = 7200) -> bool:
        """Cache document text with 2-hour TTL"""
        return self.cache.set(self.namespace, f"text:{url}", text, ttl=ttl)
    
    def get_processed_document(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached processed document"""
        return self.cache.get(self.namespace, f"processed:{url}")
    
    def set_processed_document(self, url: str, processed_doc: Dict[str, Any], ttl: int = 7200) -> bool:
        """Cache processed document"""
        return self.cache.set(self.namespace, f"processed:{url}", processed_doc, ttl=ttl)

class EmbeddingCache:
    """Specialized cache for embeddings"""
    
    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.namespace = "embeddings"
    
    def get_text_embedding(self, text: str, model: str = "default") -> Optional[List[float]]:
        """Get cached text embedding"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return self.cache.get(self.namespace, f"{model}:{text_hash}")
    
    def set_text_embedding(self, text: str, embedding: List[float], 
                          model: str = "default", ttl: int = 86400) -> bool:
        """Cache text embedding with 24-hour TTL"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return self.cache.set(self.namespace, f"{model}:{text_hash}", embedding, ttl=ttl)
    
    def get_document_embeddings(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get cached document embeddings"""
        return self.cache.get(self.namespace, f"doc:{document_id}")
    
    def set_document_embeddings(self, document_id: str, embeddings: Dict[str, Any], ttl: int = 86400) -> bool:
        """Cache document embeddings"""
        return self.cache.set(self.namespace, f"doc:{document_id}", embeddings, ttl=ttl)

class QueryCache:
    """Specialized cache for query results"""
    
    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.namespace = "queries"
    
    def get_query_result(self, document_hash: str, question: str) -> Optional[str]:
        """Get cached query result"""
        query_key = f"{document_hash}:{hashlib.sha256(question.encode()).hexdigest()[:16]}"
        return self.cache.get(self.namespace, query_key)
    
    def set_query_result(self, document_hash: str, question: str, answer: str, ttl: int = 3600) -> bool:
        """Cache query result with 1-hour TTL"""
        query_key = f"{document_hash}:{hashlib.sha256(question.encode()).hexdigest()[:16]}"
        return self.cache.set(self.namespace, query_key, answer, ttl=ttl)
    
    def get_batch_results(self, document_hash: str, questions: List[str]) -> Optional[List[str]]:
        """Get cached batch query results"""
        questions_hash = hashlib.sha256(json.dumps(questions).encode()).hexdigest()[:16]
        batch_key = f"batch:{document_hash}:{questions_hash}"
        return self.cache.get(self.namespace, batch_key)
    
    def set_batch_results(self, document_hash: str, questions: List[str], answers: List[str], ttl: int = 3600) -> bool:
        """Cache batch query results"""
        questions_hash = hashlib.sha256(json.dumps(questions).encode()).hexdigest()[:16]
        batch_key = f"batch:{document_hash}:{questions_hash}"
        return self.cache.set(self.namespace, batch_key, answers, ttl=ttl)

# Global cache instances
_global_cache = None
_document_cache = None
_embedding_cache = None
_query_cache = None

def get_cache_instance() -> IntelligentCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        redis_url = os.getenv('REDIS_URL')
        _global_cache = IntelligentCache(
            memory_cache_size=1000,
            redis_url=redis_url,
            file_cache_dir=os.getenv('CACHE_DIR', '/tmp/hackrx_cache'),
            default_ttl=3600
        )
    return _global_cache

def get_document_cache() -> DocumentCache:
    """Get document cache instance"""
    global _document_cache
    if _document_cache is None:
        _document_cache = DocumentCache(get_cache_instance())
    return _document_cache

def get_embedding_cache() -> EmbeddingCache:
    """Get embedding cache instance"""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(get_cache_instance())
    return _embedding_cache

def get_query_cache() -> QueryCache:
    """Get query cache instance"""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(get_cache_instance())
    return _query_cache