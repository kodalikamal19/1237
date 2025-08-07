"""
Optimized Query Processor with FAISS integration and advanced prompting strategies.
Designed specifically for HackRX scoring optimization with focus on unknown documents.
"""

import os
import json
import gc
from typing import List, Dict, Any, Tuple, Optional
import google.generativeai as genai
from src.training.faiss_semantic_search import AdvancedFAISSSearch, DocumentChunk, SearchResult
from src.utils.memory_manager import MemoryManager
from src.training.enhanced_fallback_processor import EnhancedFallbackProcessor
import logging
import re
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Represents a query processing result with explainability"""
    answer: str
    confidence_score: float
    relevant_clauses: List[Dict[str, Any]]
    processing_method: str
    explanation: str
    token_usage: int = 0

class OptimizedQueryProcessor:
    """
    Optimized query processor leveraging FAISS semantic search and advanced prompting.
    Specifically tuned for HackRX evaluation criteria with focus on unknown documents.
    """
    
    def __init__(self, training_data_path: str = None):
        self.memory_manager = MemoryManager()
        
        # Initialize Gemini API with optimized settings
        self.model = None
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set")
        else:
            genai.configure(api_key=api_key)
            # Use the latest and most capable model
            self.model = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        # Initialize FAISS semantic search
        self.semantic_search = AdvancedFAISSSearch(
            model_name="all-MiniLM-L6-v2",  # Fast and accurate
            dimension=384
        )
        
        # Initialize enhanced fallback processor
        self.fallback_processor = EnhancedFallbackProcessor(training_data_path)
        
        # Load training data if available
        self.training_data = []
        if training_data_path and os.path.exists(training_data_path):
            self.load_training_data(training_data_path)
        
        # Domain-specific prompt templates optimized for accuracy
        self.prompt_templates = {
            'insurance': self._get_insurance_prompt_template(),
            'legal': self._get_legal_prompt_template(),
            'hr': self._get_hr_prompt_template(),
            'compliance': self._get_compliance_prompt_template(),
            'general': self._get_general_prompt_template()
        }
        
        # Question type classifiers for optimized processing
        self.question_patterns = {
            'yes_no': [r'\b(does|is|are|can|will|has|have)\b.*\?', r'\b(yes|no)\b'],
            'numerical': [r'\b(how much|how many|what.*percent|what.*amount)\b', r'\d+'],
            'duration': [r'\b(how long|when|period|time|days?|months?|years?)\b'],
            'definition': [r'\b(what is|define|meaning of|definition)\b'],
            'procedure': [r'\b(how to|process|procedure|steps)\b'],
            'condition': [r'\b(condition|requirement|criteria|eligibility)\b']
        }
    
    def load_training_data(self, data_path: str):
        """Load and index training data for better context"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)
            
            # Index training documents in FAISS
            training_docs = []
            for doc in self.training_data:
                training_docs.append({
                    'id': doc.get('file_name', f"doc_{len(training_docs)}"),
                    'content': doc.get('content', ''),
                    'type': doc.get('document_type', 'insurance')
                })
            
            if training_docs:
                self.semantic_search.index_documents(training_docs)
                logger.info(f"Indexed {len(training_docs)} training documents")
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            self.training_data = []
    
    def classify_question_type(self, question: str) -> str:
        """Classify question type for optimized processing"""
        question_lower = question.lower()
        
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return q_type
        
        return 'general'
    
    def detect_document_domain(self, document_text: str) -> str:
        """Detect document domain for optimized processing"""
        text_lower = document_text.lower()
        
        domain_indicators = {
            'insurance': ['policy', 'coverage', 'premium', 'claim', 'insured', 'benefit', 'mediclaim'],
            'legal': ['contract', 'agreement', 'party', 'clause', 'whereas', 'liability'],
            'hr': ['employee', 'employment', 'salary', 'benefits', 'leave', 'performance'],
            'compliance': ['regulation', 'compliance', 'audit', 'standard', 'requirement']
        }
        
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score, default to insurance
        return max(domain_scores, key=domain_scores.get) if domain_scores else 'insurance'
    
    def process_query_optimized(self, document_text: str, question: str) -> QueryResult:
        """
        Process query with full optimization pipeline.
        Uses FAISS semantic search + optimized prompting + fallback strategies.
        """
        start_time = time.time()
        
        # Detect document domain and question type
        domain = self.detect_document_domain(document_text)
        question_type = self.classify_question_type(question)
        
        logger.info(f"Processing {question_type} question in {domain} domain")
        
        # Find relevant clauses using FAISS
        relevant_clauses = self.semantic_search.find_relevant_clauses(question, document_text)
        
        # Try API-based processing first
        if self.model:
            try:
                api_result = self._process_with_api(document_text, question, domain, question_type, relevant_clauses)
                if api_result and self._validate_answer_quality(api_result.answer, question):
                    api_result.processing_method = "API + FAISS"
                    api_result.relevant_clauses = relevant_clauses
                    return api_result
            except Exception as e:
                logger.warning(f"API processing failed: {str(e)}")
        
        # Fallback to enhanced pattern matching
        logger.info("Using enhanced fallback processing")
        fallback_answer = self.fallback_processor.process_query(document_text, question)
        
        result = QueryResult(
            answer=fallback_answer,
            confidence_score=0.7,  # Fallback has moderate confidence
            relevant_clauses=relevant_clauses,
            processing_method="Enhanced Fallback + FAISS",
            explanation=f"Processed using domain-aware fallback for {domain} document",
            token_usage=0
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        return result
    
    def _process_with_api(self, document_text: str, question: str, domain: str, 
                         question_type: str, relevant_clauses: List[Dict[str, Any]]) -> Optional[QueryResult]:
        """Process query using Gemini API with optimized prompting"""
        
        # Get domain-specific prompt template
        prompt_template = self.prompt_templates.get(domain, self.prompt_templates['general'])
        
        # Build context from relevant clauses
        clause_context = ""
        if relevant_clauses:
            clause_context = "\n\nMOST RELEVANT CLAUSES:\n"
            for i, clause in enumerate(relevant_clauses[:3], 1):
                clause_context += f"Clause {i} (similarity: {clause['similarity_score']:.3f}):\n"
                clause_context += f"{clause['content']}\n\n"
        
        # Optimize document text for API processing
        optimized_text = self._optimize_document_for_api(document_text, question, relevant_clauses)
        
        # Build final prompt
        prompt = prompt_template.format(
            question_type=question_type,
            domain=domain,
            document_text=optimized_text,
            clause_context=clause_context,
            question=question
        )
        
        try:
            # Generate response with optimized settings
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistency
                    top_p=0.8,       # Focused sampling
                    top_k=10,        # Limited choices for consistency
                    max_output_tokens=100,  # Concise answers
                )
            )
            
            answer = response.text.strip()
            
            # Post-process answer
            answer = self._post_process_answer(answer, question, question_type)
            
            # Calculate confidence based on answer quality
            confidence = self._calculate_confidence(answer, question, relevant_clauses)
            
            # Estimate token usage
            token_usage = len(prompt.split()) + len(answer.split())
            
            return QueryResult(
                answer=answer,
                confidence_score=confidence,
                relevant_clauses=relevant_clauses,
                processing_method="Gemini API",
                explanation=f"Processed using {domain}-optimized prompt with FAISS clause retrieval",
                token_usage=token_usage
            )
            
        except Exception as e:
            logger.error(f"API processing error: {str(e)}")
            return None
    
    def _optimize_document_for_api(self, document_text: str, question: str, 
                                  relevant_clauses: List[Dict[str, Any]]) -> str:
        """Optimize document text for API processing by focusing on relevant sections"""
        
        # If we have relevant clauses, prioritize them
        if relevant_clauses:
            # Combine relevant clause content
            relevant_content = []
            for clause in relevant_clauses[:5]:  # Top 5 clauses
                relevant_content.append(clause['content'])
            
            # Add some surrounding context
            full_relevant_text = "\n\n".join(relevant_content)
            
            # If the relevant text is sufficient, use it
            if len(full_relevant_text) > 500:
                return full_relevant_text[:8000]  # Limit for API efficiency
        
        # Otherwise, use smart truncation
        max_length = 10000  # Optimized length for API
        if len(document_text) <= max_length:
            return document_text
        
        # Try to find question-relevant sections
        question_keywords = self._extract_keywords(question)
        sections = document_text.split('\n\n')
        
        # Score sections by keyword relevance
        scored_sections = []
        for section in sections:
            score = sum(1 for keyword in question_keywords if keyword.lower() in section.lower())
            if score > 0 or len(section) > 100:  # Keep relevant or substantial sections
                scored_sections.append((score, section))
        
        # Sort by relevance and combine
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        optimized_text = ""
        
        for score, section in scored_sections:
            if len(optimized_text) + len(section) < max_length:
                optimized_text += section + "\n\n"
            else:
                break
        
        return optimized_text or document_text[:max_length]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        import re
        
        # Remove common words and extract meaningful terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common stop words to remove
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'she', 'use', 'her', 'now', 'air', 'any', 'may', 'say'}
        
        keywords = [word for word in words if word not in stop_words]
        return keywords[:10]  # Top 10 keywords
    
    def _validate_answer_quality(self, answer: str, question: str) -> bool:
        """Validate if the answer is of good quality"""
        if not answer or len(answer.strip()) < 3:
            return False
        
        # Check for common failure patterns
        failure_patterns = [
            "information not available",
            "not specified",
            "cannot be determined",
            "error processing",
            "api key not",
            "failed to",
            "unable to"
        ]
        
        answer_lower = answer.lower()
        for pattern in failure_patterns:
            if pattern in answer_lower:
                return False
        
        # Check for reasonable answer length
        if len(answer) > 500:  # Too long
            return False
        
        return True
    
    def _calculate_confidence(self, answer: str, question: str, 
                            relevant_clauses: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the answer"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if we have relevant clauses
        if relevant_clauses:
            avg_similarity = sum(clause['similarity_score'] for clause in relevant_clauses) / len(relevant_clauses)
            confidence += avg_similarity * 0.3
        
        # Boost confidence for specific answers
        if any(char.isdigit() for char in answer):  # Contains numbers
            confidence += 0.1
        
        if len(answer.split()) >= 3:  # Reasonable length
            confidence += 0.1
        
        # Check for definitive language
        definitive_words = ['yes', 'no', 'exactly', 'specifically', 'precisely']
        if any(word in answer.lower() for word in definitive_words):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _post_process_answer(self, answer: str, question: str, question_type: str) -> str:
        """Post-process answer based on question type"""
        answer = answer.strip()
        
        # Ensure proper capitalization
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Remove common AI hedging for direct answers
        hedging_phrases = [
            "Based on the document, ",
            "According to the document, ",
            "The document states that ",
            "The document indicates that ",
            "It appears that ",
            "The answer is ",
        ]
        
        for phrase in hedging_phrases:
            if answer.startswith(phrase):
                answer = answer[len(phrase):]
                if answer and answer[0].islower():
                    answer = answer[0].upper() + answer[1:]
                break
        
        # Ensure proper ending punctuation
        if answer and not answer.endswith(('.', '!', '?')):
            answer += "."
        
        return answer
    
    def batch_process_queries(self, document_text: str, questions: List[str]) -> List[str]:
        """Process multiple queries with optimization for batch processing"""
        logger.info(f"Processing {len(questions)} queries in optimized batch mode")
        
        # Pre-analyze document once
        domain = self.detect_document_domain(document_text)
        logger.info(f"Document domain detected: {domain}")
        
        answers = []
        
        # Process questions with intelligent batching
        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
                
                result = self.process_query_optimized(document_text, question)
                answers.append(result.answer)
                
                # Log processing details
                logger.info(f"Answer: {result.answer[:100]}...")
                logger.info(f"Method: {result.processing_method}, Confidence: {result.confidence_score:.2f}")
                
                # Memory management
                if i % 3 == 0:
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        logger.info("Batch processing completed")
        return answers
    
    def _get_insurance_prompt_template(self) -> str:
        """Optimized prompt template for insurance documents"""
        return """You are an expert insurance policy analyst. Analyze the insurance document and answer the question with maximum accuracy and precision.

CRITICAL INSTRUCTIONS:
- Extract EXACT information from the document
- For numerical values, provide the precise number mentioned
- For yes/no questions, give a definitive answer based on document content
- For conditions, list the specific requirements mentioned
- Be concise but complete - no unnecessary explanations

DOCUMENT TYPE: {domain}
QUESTION TYPE: {question_type}

{clause_context}

FULL DOCUMENT:
{document_text}

QUESTION: {question}

PRECISE ANSWER (be direct and factual):"""
    
    def _get_legal_prompt_template(self) -> str:
        """Optimized prompt template for legal documents"""
        return """You are a legal document expert. Analyze the legal document and provide a precise answer based solely on the document content.

CRITICAL INSTRUCTIONS:
- Quote exact terms and conditions when relevant
- For contractual questions, identify specific clauses
- For obligations, state exactly what is required
- For rights, specify what is granted or restricted
- Be legally precise and factual

DOCUMENT TYPE: {domain}
QUESTION TYPE: {question_type}

{clause_context}

LEGAL DOCUMENT:
{document_text}

QUESTION: {question}

LEGAL ANSWER (precise and fact-based):"""
    
    def _get_hr_prompt_template(self) -> str:
        """Optimized prompt template for HR documents"""
        return """You are an HR policy expert. Analyze the HR document and provide accurate information about policies, benefits, or procedures.

CRITICAL INSTRUCTIONS:
- Extract specific policy details
- For benefit questions, provide exact amounts or percentages
- For procedure questions, outline the specific steps
- For eligibility, state the precise criteria
- Be clear and actionable

DOCUMENT TYPE: {domain}
QUESTION TYPE: {question_type}

{clause_context}

HR DOCUMENT:
{document_text}

QUESTION: {question}

HR ANSWER (clear and specific):"""
    
    def _get_compliance_prompt_template(self) -> str:
        """Optimized prompt template for compliance documents"""
        return """You are a compliance expert. Analyze the compliance document and provide accurate information about regulations, requirements, or standards.

CRITICAL INSTRUCTIONS:
- Identify specific regulatory requirements
- For compliance questions, state exact obligations
- For audit questions, specify what is required
- For violations, indicate consequences mentioned
- Be regulatory-precise

DOCUMENT TYPE: {domain}
QUESTION TYPE: {question_type}

{clause_context}

COMPLIANCE DOCUMENT:
{document_text}

QUESTION: {question}

COMPLIANCE ANSWER (regulation-specific):"""
    
    def _get_general_prompt_template(self) -> str:
        """General optimized prompt template"""
        return """You are a document analysis expert. Analyze the document and answer the question with maximum accuracy based on the document content.

CRITICAL INSTRUCTIONS:
- Extract information directly from the document
- Provide specific details when available
- For numerical questions, give exact numbers
- For procedural questions, outline specific steps
- Be factual and concise

DOCUMENT TYPE: {domain}
QUESTION TYPE: {question_type}

{clause_context}

DOCUMENT:
{document_text}

QUESTION: {question}

ANSWER (direct and factual):"""
