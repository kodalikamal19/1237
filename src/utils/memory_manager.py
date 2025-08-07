import gc
import psutil
import os
from functools import wraps
from typing import Callable, Any

class MemoryManager:
    """Memory management utilities for efficient processing"""
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage statistics"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection to free memory"""
        gc.collect()
        gc.collect()  # Call twice for better cleanup
        gc.collect()
    
    @staticmethod
    def memory_limit_check(max_memory_mb: int = 512) -> bool:
        """Check if memory usage is within limits"""
        memory_usage = MemoryManager.get_memory_usage()
        return memory_usage['rss_mb'] < max_memory_mb
    
    @staticmethod
    def cleanup_decorator(func: Callable) -> Callable:
        """Decorator to automatically cleanup memory after function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                MemoryManager.force_garbage_collection()
        return wrapper

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class StreamingProcessor:
    """Process data in streaming fashion to minimize memory usage"""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.memory_manager = MemoryManager()
    
    def process_with_memory_check(self, data_processor: Callable, data: Any) -> Any:
        """Process data with memory monitoring"""
        # Check memory before processing
        if not self.memory_manager.memory_limit_check(self.max_memory_mb):
            self.memory_manager.force_garbage_collection()
            
            # If still over limit, raise error
            if not self.memory_manager.memory_limit_check(self.max_memory_mb):
                raise MemoryError(f"Memory usage exceeds {self.max_memory_mb}MB limit")
        
        try:
            result = data_processor(data)
            return result
        finally:
            # Cleanup after processing
            self.memory_manager.force_garbage_collection()

