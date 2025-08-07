"""
Performance optimizer for improving runtime and efficiency
"""
import asyncio
import concurrent.futures
import time
from typing import List, Dict, Any, Callable
import threading
import queue
import gc
from functools import wraps
from src.utils.memory_manager import MemoryManager

class PerformanceOptimizer:
    """Optimize performance and runtime of the HackRX API"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 100
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'avg_response_time': 0
        }
    
    def cache_response(self, key_func: Callable = None):
        """Decorator to cache responses based on input"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = str(hash(str(args) + str(kwargs)))
                
                # Check cache
                with self.cache_lock:
                    if cache_key in self.cache:
                        self.stats['cache_hits'] += 1
                        return self.cache[cache_key]
                    
                    self.stats['cache_misses'] += 1
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Store in cache
                with self.cache_lock:
                    if len(self.cache) >= self.max_cache_size:
                        # Remove oldest entry
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                    
                    self.cache[cache_key] = result
                
                return result
            return wrapper
        return decorator
    
    def parallel_process_queries(self, processor, document_text: str, questions: List[str], max_workers: int = 3) -> List[str]:
        """Process queries in parallel for better performance"""
        if len(questions) <= 2:
            # For small number of questions, sequential is faster
            return processor.batch_process_queries(document_text, questions)
        
        # Split questions into chunks for parallel processing
        chunk_size = max(1, len(questions) // max_workers)
        question_chunks = [questions[i:i + chunk_size] for i in range(0, len(questions), chunk_size)]
        
        answers = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_chunk = {}
            for chunk in question_chunks:
                if chunk:  # Only submit non-empty chunks
                    future = executor.submit(processor.batch_process_queries, document_text, chunk)
                    future_to_chunk[future] = chunk
            
            # Collect results in order
            chunk_results = {}
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    chunk_results[question_chunks.index(chunk)] = result
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    # Fallback to error messages
                    chunk_results[question_chunks.index(chunk)] = [f"Error: {str(e)}" for _ in chunk]
        
        # Combine results in original order
        for i in range(len(question_chunks)):
            if i in chunk_results:
                answers.extend(chunk_results[i])
        
        return answers
    
    def optimize_text_processing(self, text: str) -> str:
        """Optimize text processing for better performance"""
        # Quick preprocessing optimizations
        if not text or len(text) < 100:
            return text
        
        # Remove excessive whitespace more efficiently
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Optimize for common patterns in insurance documents
        text = self._optimize_insurance_text(text)
        
        return text.strip()
    
    def _optimize_insurance_text(self, text: str) -> str:
        """Optimize text specifically for insurance documents"""
        import re
        
        # Common insurance document optimizations
        replacements = {
            r'\bRs\.?\s*': '₹',  # Standardize currency
            r'\bINR\s*': '₹',
            r'\brupees?\b': '₹',
            r'\bpolicy\s+holder\b': 'policyholder',
            r'\bsum\s+insured\b': 'sum insured',
            r'\bclaim\s+settlement\b': 'claim settlement',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def batch_optimize_memory(self):
        """Optimize memory usage in batch"""
        # Force garbage collection
        gc.collect()
        
        # Clear cache if memory usage is high
        memory_usage = self.memory_manager.get_memory_usage()
        if memory_usage['percent'] > 70:  # If using more than 70% memory
            with self.cache_lock:
                self.cache.clear()
                print("Cache cleared due to high memory usage")
        
        return memory_usage
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = 0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache),
            'total_requests': self.stats['total_requests'],
            'avg_response_time': self.stats['avg_response_time'],
            'memory_usage': self.memory_manager.get_memory_usage()
        }
    
    def time_function(self, func: Callable) -> Callable:
        """Decorator to time function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Update stats
            self.stats['total_requests'] += 1
            if self.stats['avg_response_time'] == 0:
                self.stats['avg_response_time'] = execution_time
            else:
                # Running average
                self.stats['avg_response_time'] = (
                    (self.stats['avg_response_time'] * (self.stats['total_requests'] - 1) + execution_time) 
                    / self.stats['total_requests']
                )
            
            print(f"Function {func.__name__} executed in {execution_time:.2f}s")
            return result
        return wrapper

class AsyncQueryProcessor:
    """Asynchronous query processor for better concurrency"""
    
    def __init__(self, base_processor):
        self.base_processor = base_processor
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
    
    async def process_query_async(self, document_text: str, question: str) -> str:
        """Process single query asynchronously"""
        async with self.semaphore:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                self.base_processor.process_query_with_enhancement,
                document_text,
                question
            )
    
    async def process_queries_async(self, document_text: str, questions: List[str]) -> List[str]:
        """Process multiple queries asynchronously"""
        tasks = []
        for question in questions:
            task = self.process_query_async(document_text, question)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        answers = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                answers.append(f"Error processing question: {str(result)}")
            else:
                answers.append(result)
        
        return answers

class ResponseCompressor:
    """Compress responses to reduce bandwidth"""
    
    @staticmethod
    def compress_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress response data"""
        if 'answers' in response_data:
            # Remove redundant phrases
            compressed_answers = []
            for answer in response_data['answers']:
                compressed_answer = ResponseCompressor._compress_text(answer)
                compressed_answers.append(compressed_answer)
            
            response_data['answers'] = compressed_answers
        
        return response_data
    
    @staticmethod
    def _compress_text(text: str) -> str:
        """Compress individual text responses"""
        if not text or len(text) < 50:
            return text
        
        # Remove redundant phrases
        redundant_phrases = [
            "Based on the document provided, ",
            "According to the information in the document, ",
            "The document states that ",
            "As mentioned in the document, "
        ]
        
        for phrase in redundant_phrases:
            if text.startswith(phrase):
                text = text[len(phrase):]
                break
        
        # Ensure proper capitalization
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Further truncate if still too long for a summarized answer (e.g., max 250 characters)
        max_summary_length = 250
        if len(text) > max_summary_length:
            # Try to cut at sentence boundary
            last_period = text.rfind(".", 0, max_summary_length)
            if last_period != -1 and last_period > max_summary_length * 0.7:
                text = text[:last_period + 1]
            else:
                text = text[:max_summary_length] + "..."
        
        return text.strip()

