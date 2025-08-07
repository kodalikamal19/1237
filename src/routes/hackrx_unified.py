import os
import io
import gc
import tempfile
from typing import List, Dict, Any
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import requests
import pypdf
import google.generativeai as genai
from src.utils.memory_manager import MemoryManager, chunk_text, StreamingProcessor
from src.training.enhanced_model import EnhancedQueryProcessor
from src.training.optimized_query_processor import OptimizedQueryProcessor
from src.utils.intelligent_cache import get_document_cache, get_query_cache, get_embedding_cache
from src.training.scoring_optimizer import HackRXScoringOptimizer
import hashlib

# Import OCR libraries for image-based PDFs
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ Warning: OCR libraries not available. Image-based PDFs may not be processed.")

hackrx_unified_bp = Blueprint("hackrx_unified", __name__)

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("⚠️ Warning: GOOGLE_API_KEY environment variable not set")
else:
    genai.configure(api_key=api_key)

class EnhancedPDFProcessor:
    """Enhanced memory-efficient PDF processor with OCR support for image-based PDFs"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
    
    @MemoryManager.cleanup_decorator
    def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL with improved error handling and validation"""
        try:
            # Validate URL format
            if not url or not isinstance(url, str):
                raise ValueError("Invalid URL provided")
            
            # Clean URL (remove extra spaces, etc.)
            url = url.strip()
            
            # Set comprehensive headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/pdf,application/octet-stream,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            print(f"Downloading PDF from: {url}")
            
            # Use session for better connection handling
            session = requests.Session()
            session.headers.update(headers)
            
            response = session.get(url, stream=True, timeout=120)  # Increased timeout
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            print(f"Content-Type: {content_type}")
            
            # Check content length to avoid downloading huge files
            content_length = response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                print(f"PDF size: {size_mb:.2f} MB")
                if int(content_length) > 100 * 1024 * 1024:  # 100MB limit
                    raise ValueError(f"PDF file too large ({size_mb:.2f}MB > 100MB)")
            
            # Read in chunks to manage memory
            pdf_data = io.BytesIO()
            total_size = 0
            max_size = 100 * 1024 * 1024  # 100MB limit
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    total_size += len(chunk)
                    if total_size > max_size:
                        pdf_data.close()
                        session.close()
                        raise ValueError("PDF file too large (>100MB)")
                    
                    pdf_data.write(chunk)
                    
                    # Check memory usage periodically
                    if total_size % (5 * 1024 * 1024) == 0:  # Every 5MB
                        if not self.memory_manager.memory_limit_check(500):  # 500MB limit
                            pdf_data.close()
                            session.close()
                            raise MemoryError("Memory limit exceeded during download")
            
            pdf_bytes = pdf_data.getvalue()
            pdf_data.close()
            session.close()
            
            print(f"Successfully downloaded PDF: {len(pdf_bytes)} bytes")
            return pdf_bytes
            
        except requests.RequestException as e:
            raise Exception(f"Failed to download PDF: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing PDF download: {str(e)}")
    
    @MemoryManager.cleanup_decorator
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Enhanced text extraction with OCR fallback for image-based PDFs"""
        try:
            # First, try standard text extraction
            text = self._extract_text_standard(pdf_bytes)
            
            # If no text extracted or very little text, try OCR
            if not text or len(text.strip()) < 100:
                print("Standard text extraction yielded little content, trying OCR...")
                if OCR_AVAILABLE:
                    ocr_text = self._extract_text_ocr(pdf_bytes)
                    if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                        print("OCR extraction successful")
                        return ocr_text
                    else:
                        print("OCR extraction failed or yielded less content")
                else:
                    print("OCR libraries not available")
            
            if not text or len(text.strip()) < 10:
                raise Exception("No text could be extracted from the PDF. The PDF may be image-based or corrupted.")
            
            return text
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_text_standard(self, pdf_bytes: bytes) -> str:
        """Standard text extraction using pypdf"""
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            reader = pypdf.PdfReader(pdf_stream)
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                print("PDF is encrypted, attempting to decrypt...")
                try:
                    reader.decrypt("")  # Try empty password
                except:
                    raise Exception("PDF is password-protected and cannot be decrypted")
            
            # Check number of pages
            num_pages = len(reader.pages)
            print(f"PDF has {num_pages} pages")
            
            if num_pages > 1000:  # Limit pages to prevent memory issues
                print(f"Warning: PDF has {num_pages} pages, processing first 1000 only")
                num_pages = 1000
            
            text_parts = []
            for page_num in range(min(num_pages, len(reader.pages))):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        cleaned_text = self._clean_extracted_text(page_text.strip())
                        if cleaned_text:
                            text_parts.append(cleaned_text)
                    
                    # Clear page from memory
                    del page
                    
                    # Garbage collect every 20 pages and check memory
                    if page_num % 20 == 0:
                        gc.collect()
                        if not self.memory_manager.memory_limit_check(500):
                            print(f"Memory limit reached at page {page_num}, stopping extraction")
                            break
                        
                except Exception as e:
                    print(f"Warning: Failed to extract text from page {page_num + 1}: {str(e)}")
                    continue
            
            # Clean up
            pdf_stream.close()
            del reader
            gc.collect()
            
            if not text_parts:
                return ""
            
            # Join text parts efficiently
            full_text = "\n\n".join(text_parts)
            del text_parts
            
            # Enhanced text post-processing
            full_text = self._post_process_text(full_text)
            
            # Limit text length to prevent memory issues
            max_text_length = 300000  # 300KB
            if len(full_text) > max_text_length:
                full_text = full_text[:max_text_length] + "\n\n[Document truncated due to length]"
            
            return full_text
            
        except Exception as e:
            print(f"Standard text extraction failed: {str(e)}")
            return ""
    
    def _extract_text_ocr(self, pdf_bytes: bytes) -> str:
        """OCR-based text extraction for image-based PDFs"""
        if not OCR_AVAILABLE:
            return ""
        
        try:
            print("Starting OCR text extraction...")
            
            # Convert PDF to images
            images = convert_from_bytes(
                pdf_bytes,
                dpi=200,  # Good balance between quality and speed
                first_page=1,
                last_page=50,  # Limit pages for OCR to prevent timeout
                fmt='jpeg',
                thread_count=2
            )
            
            print(f"Converted PDF to {len(images)} images")
            
            text_parts = []
            for i, image in enumerate(images):
                try:
                    # Use OCR to extract text from image
                    page_text = pytesseract.image_to_string(
                        image,
                        config='--psm 6 --oem 3'  # Optimized for documents
                    )
                    
                    if page_text and page_text.strip():
                        cleaned_text = self._clean_extracted_text(page_text.strip())
                        if cleaned_text:
                            text_parts.append(cleaned_text)
                    
                    # Clear image from memory
                    image.close()
                    del image
                    
                    print(f"Processed OCR page {i+1}/{len(images)}")
                    
                    # Check memory usage
                    if i % 5 == 0:  # Every 5 pages
                        gc.collect()
                        if not self.memory_manager.memory_limit_check(600):
                            print(f"Memory limit reached during OCR at page {i+1}")
                            break
                    
                except Exception as e:
                    print(f"OCR failed for page {i+1}: {str(e)}")
                    continue
            
            # Clean up remaining images
            for img in images:
                try:
                    img.close()
                except:
                    pass
            del images
            gc.collect()
            
            if not text_parts:
                return ""
            
            # Join text parts
            full_text = "\n\n".join(text_parts)
            del text_parts
            
            # Post-process OCR text
            full_text = self._post_process_ocr_text(full_text)
            
            print(f"OCR extraction completed: {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            print(f"OCR text extraction failed: {str(e)}")
            return ""
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text from PDF"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s.,:;!?()-[\]{}"\'/\\@#$%^&*+=<>~`|]', '', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = text.replace('0', 'O')  # In some contexts
        
        return text.strip()
    
    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text for better readability"""
        import re
        
        # Fix line breaks and spacing
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1\n\n\2', text)
        
        # Fix common formatting issues
        text = re.sub(r'\s+([.,:;!?])', r'\1', text)
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)
        
        return text
    
    def _post_process_ocr_text(self, text: str) -> str:
        """Post-process OCR text with additional cleaning"""
        import re
        
        # Standard post-processing
        text = self._post_process_text(text)
        
        # OCR-specific fixes
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add spaces between words
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Space between letters and numbers
        
        # Fix common OCR character mistakes
        replacements = {
            'rn': 'm',
            'vv': 'w',
            'VV': 'W',
            '|': 'I',
            '0': 'O',  # Context-dependent
            '5': 'S',  # Context-dependent
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

# Initialize optimized processor with training data and scoring optimizer
training_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src", "training", "processed_training_data.json")
optimized_processor = None
scoring_optimizer = None

try:
    if os.path.exists(training_data_path):
        optimized_processor = OptimizedQueryProcessor(training_data_path)
        scoring_optimizer = HackRXScoringOptimizer(optimized_processor)
        print("✅ Optimized processor and scoring optimizer initialized with training data and FAISS indexing")
    else:
        optimized_processor = OptimizedQueryProcessor()
        scoring_optimizer = HackRXScoringOptimizer(optimized_processor)
        print("⚠️ Optimized processor and scoring optimizer initialized without training data")
except Exception as e:
    print(f"⚠️ Error initializing optimized processor: {str(e)}")
    # Fallback to enhanced processor
    try:
        if os.path.exists(training_data_path):
            optimized_processor = EnhancedQueryProcessor(training_data_path)
            print("✅ Fallback: Enhanced processor initialized")
        else:
            optimized_processor = EnhancedQueryProcessor()
            print("⚠️ Fallback: Enhanced processor initialized without training data")
        scoring_optimizer = None  # No scoring optimizer available
    except Exception as e2:
        print(f"⚠️ Error initializing fallback processor: {str(e2)}")
        optimized_processor = EnhancedQueryProcessor()
        scoring_optimizer = None

@hackrx_unified_bp.route('/run', methods=['POST'])
@cross_origin()
def hackrx_unified_run():
    """Unified HackRX API endpoint with enhanced error handling, caching, and PDF processing"""
    memory_manager = MemoryManager()
    
    # Initialize caches
    document_cache = get_document_cache()
    query_cache = get_query_cache()
    
    try:
        # Log initial memory usage
        initial_memory = memory_manager.get_memory_usage()
        print(f"Initial memory usage: {initial_memory['rss_mb']:.2f}MB")
        
        # Enhanced request validation
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'details': 'Please ensure your request has Content-Type: application/json header'
            }), 400
        
        try:
            data = request.get_json()
        except Exception as e:
            return jsonify({
                'error': 'Invalid JSON data',
                'details': f'JSON parsing failed: {str(e)}'
            }), 400
        
        if not data:
            return jsonify({
                'error': 'Empty request body',
                'details': 'Request body cannot be empty'
            }), 400
        
        # Validate required fields with detailed error messages
        if 'documents' not in data:
            return jsonify({
                'error': 'Missing required field: documents',
                'details': 'The "documents" field is required and should contain a PDF URL'
            }), 400
        
        if 'questions' not in data:
            return jsonify({
                'error': 'Missing required field: questions',
                'details': 'The "questions" field is required and should contain a list of questions'
            }), 400
        
        documents_url = data['documents']
        questions = data['questions']
        
        # Enhanced input validation
        if not isinstance(documents_url, str) or not documents_url.strip():
            return jsonify({
                'error': 'Invalid documents field',
                'details': 'documents must be a valid URL string'
            }), 400
        
        if not isinstance(questions, list) or len(questions) == 0:
            return jsonify({
                'error': 'Invalid questions field',
                'details': 'questions must be a non-empty list of strings'
            }), 400
        
        if len(questions) > 50:  # Reasonable limit
            return jsonify({
                'error': 'Too many questions',
                'details': f'Maximum 50 questions allowed, received {len(questions)}'
            }), 400
        
        # Validate each question
        for i, question in enumerate(questions):
            if not isinstance(question, str) or not question.strip():
                return jsonify({
                    'error': f'Invalid question at index {i}',
                    'details': f'Question {i+1} must be a non-empty string'
                }), 400
            if len(question) > 2000:
                return jsonify({
                    'error': f'Question {i+1} too long',
                    'details': f'Maximum 2000 characters allowed, question has {len(question)}'
                }), 400
        
        # Generate document hash for caching
        document_hash = hashlib.sha256(documents_url.encode()).hexdigest()[:16]
        
        # Check for cached batch results first
        cached_answers = query_cache.get_batch_results(document_hash, questions)
        if cached_answers:
            print("✅ Returning cached batch results")
            return jsonify({'answers': cached_answers}), 200
        
        # Process PDF with enhanced error handling and caching
        try:
            print("Starting enhanced PDF processing with caching...")
            
            # Check for cached document text
            document_text = document_cache.get_document_text(documents_url)
            
            if document_text:
                print("✅ Using cached document text")
            else:
                print("📥 Processing document from URL...")
                pdf_processor = EnhancedPDFProcessor()
                
                # Download PDF
                try:
                    pdf_bytes = pdf_processor.download_pdf(documents_url)
                except Exception as e:
                    return jsonify({
                        'error': 'PDF download failed',
                        'details': str(e),
                        'url': documents_url
                    }), 400
                
                # Check memory after download
                memory_after_download = memory_manager.get_memory_usage()
                print(f"Memory after PDF download: {memory_after_download['rss_mb']:.2f}MB")
                
                # Extract text
                try:
                    document_text = pdf_processor.extract_text_from_pdf(pdf_bytes)
                except Exception as e:
                    return jsonify({
                        'error': 'PDF text extraction failed',
                        'details': str(e),
                        'suggestions': [
                            'Ensure the PDF is not password-protected',
                            'Check if the PDF contains readable text (not just images)',
                            'Try a different PDF file'
                        ]
                    }), 400
                
                # Clear PDF bytes from memory immediately
                del pdf_bytes
                gc.collect()
                
                # Cache the document text
                document_cache.set_document_text(documents_url, document_text, ttl=7200)  # 2 hours
            
            print(f"Extracted text length: {len(document_text)} characters")
            
            if len(document_text.strip()) < 50:
                return jsonify({
                    'error': 'Insufficient text extracted',
                    'details': 'The PDF appears to contain very little readable text',
                    'extracted_length': len(document_text),
                    'suggestions': [
                        'Check if the PDF is image-based (scanned document)',
                        'Ensure the PDF is not corrupted',
                        'Try a different PDF file'
                    ]
                }), 400
            
        except Exception as e:
            return jsonify({
                'error': 'PDF processing failed',
                'details': str(e)
            }), 400
        
        # Process queries with scoring-optimized processor and caching
        try:
            print("Starting HackRX scoring-optimized query processing...")
            
            # Check for individual cached answers first
            cached_individual = []
            uncached_questions = []
            uncached_indices = []
            
            for i, question in enumerate(questions):
                cached_answer = query_cache.get_query_result(document_hash, question)
                if cached_answer:
                    cached_individual.append((i, cached_answer))
                else:
                    uncached_questions.append(question)
                    uncached_indices.append(i)
            
            print(f"Found {len(cached_individual)} cached answers, processing {len(uncached_questions)} new questions")
            
            # Process uncached questions with scoring optimization
            new_answers = []
            if uncached_questions:
                if scoring_optimizer:
                    # Use HackRX scoring optimizer for maximum performance
                    print("🎯 Using HackRX Scoring Optimizer for maximum accuracy on unknown documents")
                    new_answers = scoring_optimizer.process_with_scoring_optimization(document_text, uncached_questions)
                    
                    # Get scoring analysis
                    try:
                        scoring_analysis = scoring_optimizer.get_scoring_analysis(document_text, uncached_questions, new_answers)
                        print(f"📊 Scoring Analysis: {scoring_analysis['document_type']} document, "
                              f"efficiency: {scoring_analysis['score_efficiency']:.1%}, "
                              f"high priority: {scoring_analysis['high_priority_questions']}")
                    except Exception as e:
                        print(f"Warning: Scoring analysis failed: {str(e)}")
                
                elif optimized_processor:
                    # Fallback to standard optimized processing
                    print("⚡ Using standard optimized processing")
                    if hasattr(optimized_processor, 'batch_process_queries'):
                        new_answers = optimized_processor.batch_process_queries(document_text, uncached_questions)
                    else:
                        # Individual processing fallback
                        new_answers = []
                        for question in uncached_questions:
                            try:
                                if hasattr(optimized_processor, 'process_query_optimized'):
                                    result = optimized_processor.process_query_optimized(document_text, question)
                                    answer = result.answer if hasattr(result, 'answer') else result
                                else:
                                    answer = optimized_processor.process_query_hybrid(document_text, question)
                                new_answers.append(answer)
                            except Exception as e:
                                print(f"Error processing individual question: {str(e)}")
                                new_answers.append(f"Error processing question: {str(e)}")
                else:
                    return jsonify({
                        'error': 'Query processor not available',
                        'details': 'Neither optimized processor nor scoring optimizer available'
                    }), 500
                
                # Cache new answers
                for question, answer in zip(uncached_questions, new_answers):
                    query_cache.set_query_result(document_hash, question, answer, ttl=3600)  # 1 hour
            
            # Combine cached and new answers in correct order
            final_answers = [''] * len(questions)
            
            # Place cached answers
            for i, answer in cached_individual:
                final_answers[i] = answer
            
            # Place new answers
            for idx, answer in zip(uncached_indices, new_answers):
                final_answers[idx] = answer
            
            answers = final_answers
            
            # Cache batch results for future requests
            if answers:
                query_cache.set_batch_results(document_hash, questions, answers, ttl=3600)
            
            # Clear document text from memory
            del document_text
            gc.collect()
            
            print("HackRX scoring-optimized processing completed successfully")
            
        except Exception as e:
            return jsonify({
                'error': 'Query processing failed',
                'details': str(e)
            }), 500
        
        # Validate response
        if len(answers) != len(questions):
            return jsonify({
                'error': 'Response validation failed',
                'details': f'Mismatch between questions ({len(questions)}) and answers ({len(answers)})'
            }), 500
        
        # Final memory check
        final_memory = memory_manager.get_memory_usage()
        print(f"Final memory usage: {final_memory['rss_mb']:.2f}MB")
        
        # Log cache statistics
        cache_stats = query_cache.cache.get_stats()
        print(f"Cache hit rate: {cache_stats['overall_hit_rate']:.2%}")
        
        return jsonify({
            'answers': answers
        }), 200
        

        
    except MemoryError as e:
        memory_manager.force_garbage_collection()
        return jsonify({
            'error': 'Memory limit exceeded',
            'details': str(e),
            'suggestions': [
                'Try with a smaller PDF file',
                'Reduce the number of questions',
                'Contact support if the issue persists'
            ]
        }), 507
        
    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"Unexpected error in hackrx_unified_run: {str(e)}")
        memory_manager.force_garbage_collection()
        return jsonify({
            'error': 'Internal server error',
            'details': 'An unexpected error occurred while processing your request'
        }), 500

@hackrx_unified_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check_unified():
    """Enhanced health check endpoint with cache statistics"""
    memory_manager = MemoryManager()
    memory_usage = memory_manager.get_memory_usage()
    
    # Check API key availability
    api_key_status = "configured" if os.getenv("GOOGLE_API_KEY") else "missing"
    
    # Check OCR availability
    ocr_status = "available" if OCR_AVAILABLE else "unavailable"
    
    # Get cache statistics
    try:
        cache_stats = get_query_cache().cache.get_stats()
    except:
        cache_stats = {"error": "Cache not available"}
    
    return jsonify({
        'status': 'healthy',
        'service': 'HackRX Unified API',
        'version': '4.0.0',  # Updated version
        'model': 'gemini-1.5-pro-latest',
        'memory_usage_mb': round(memory_usage['rss_mb'], 2),
        'memory_percent': round(memory_usage['percent'], 2),
        'api_key_status': api_key_status,
        'ocr_status': ocr_status,
        'cache_stats': cache_stats,
        'enhanced_features': [
            'HackRX Scoring Optimization',
            'FAISS semantic search',
            'Multi-layer intelligent caching',
            'OCR support for image-based PDFs',
            'Enhanced error handling',
            'Improved text extraction',
            'Memory optimization',
            'Document similarity matching',
            'Training data integration',
            'Domain-aware processing',
            'Unknown document prioritization',
            'Question complexity analysis'
        ],
        'scoring_optimizer_available': scoring_optimizer is not None,
        'training_documents': len(optimized_processor.training_data) if optimized_processor else 0
    }), 200

@hackrx_unified_bp.route('/test', methods=['POST'])
@cross_origin()
def test_endpoint():
    """Test endpoint for debugging"""
    try:
        data = request.get_json()
        return jsonify({
            'status': 'success',
            'received_data': data,
            'data_type': type(data).__name__,
            'has_documents': 'documents' in data if data else False,
            'has_questions': 'questions' in data if data else False,
            'questions_count': len(data.get('questions', [])) if data else 0
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 400