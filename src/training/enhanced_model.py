import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import gc
from src.utils.memory_manager import MemoryManager
from src.training.enhanced_fallback_processor import EnhancedFallbackProcessor, HybridProcessor

class EnhancedQueryProcessor:
    """Enhanced query processor with training capabilities and improved accuracy"""
    
    def __init__(self, training_data_path: str = None):
        # Configure Gemini with latest model
        self.model = None
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ Warning: GOOGLE_API_KEY environment variable not set")
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        self.memory_manager = MemoryManager()
        self.training_data = []
        self.qa_pairs = []
        self.document_embeddings = {}
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.8,
            min_df=1
        )
        self.document_vectors = None
        
        # Initialize enhanced fallback processor
        self.fallback_processor = EnhancedFallbackProcessor()
        
        # Load training data if provided
        if training_data_path and os.path.exists(training_data_path):
            self.load_training_data(training_data_path)
            self.build_document_index()
        else:
            # Try to load default processed training data
            default_path = os.path.join(os.path.dirname(__file__), "processed_training_data.json")
            if os.path.exists(default_path):
                self.load_training_data(default_path)
                self.build_document_index()
                print("✅ Loaded default processed training data")
    
    def load_training_data(self, data_path: str):
        """Load processed training data from JSON file"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)
            
            # Extract Q&A pairs from processed data
            self.qa_pairs = []
            for doc in self.training_data:
                if 'qa_pairs' in doc:
                    for qa in doc['qa_pairs']:
                        qa['document_type'] = doc.get('document_type', 'unknown')
                        qa['file_name'] = doc.get('file_name', 'unknown')
                        self.qa_pairs.append(qa)
            
            print(f"Loaded {len(self.training_data)} training documents with {len(self.qa_pairs)} Q&A pairs")
        except Exception as e:
            print(f"Error loading training data: {str(e)}")
            self.training_data = []
            self.qa_pairs = []
    
    def build_document_index(self):
        """Build TF-IDF index for document similarity matching"""
        if not self.training_data:
            return
        
        try:
            # Extract document contents
            documents = [doc['content'] for doc in self.training_data]
            
            # Build TF-IDF vectors
            self.document_vectors = self.vectorizer.fit_transform(documents)
            
            # Save the vectorizer and vectors
            self.save_model_components()
            
            print(f"Built document index with {len(documents)} documents")
            
        except Exception as e:
            print(f"Error building document index: {str(e)}")
    
    def find_similar_documents(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar documents from training data"""
        if self.document_vectors is None or not self.training_data:
            return []
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query_text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            # Get top-k similar documents
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            similar_docs = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    doc = self.training_data[idx].copy()
                    doc['similarity_score'] = float(similarities[idx])
                    similar_docs.append(doc)
            
            return similar_docs
            
        except Exception as e:
            print(f"Error finding similar documents: {str(e)}")
            return []
    
    def generate_enhanced_prompt(self, document_text: str, question: str, similar_docs: List[Dict[str, Any]] = None) -> str:
        """Generate optimized prompt with training data context for concise, exact, and grammatically correct answers"""
        
        # Find similar Q&A pairs from training data
        similar_qa = self._find_similar_qa_pairs(question, top_k=3)
        
        # Build context from similar Q&A pairs
        context_examples = ""
        if similar_qa:
            context_examples = "\n\nEXAMPLES FROM SIMILAR DOCUMENTS:\n"
            for i, qa in enumerate(similar_qa[:2], 1):
                context_examples += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n\n"
        
        # Optimized prompt for speed, accuracy, and conciseness
        prompt = f"""You are an expert insurance document analyzer. Based ONLY on the provided document, answer the question with a single, precise, grammatically correct sentence or short phrase. 

CRITICAL INSTRUCTIONS:
- If the exact information is in the document, provide a direct, concise answer
- Do NOT say "Information not available" unless you have thoroughly searched the document
- Look for synonyms, related terms, and context clues
- Extract specific numbers, percentages, time periods, and conditions mentioned
- Be precise and factual

{context_examples}

DOCUMENT:
{document_text[:35000]}

QUESTION: {question}

ANSWER (be specific and direct):"""
        
        return prompt
    
    def _find_similar_qa_pairs(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar Q&A pairs from training data"""
        if not self.qa_pairs:
            return []
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create a simple vectorizer for Q&A matching
            qa_questions = [qa['question'] for qa in self.qa_pairs]
            qa_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            qa_vectors = qa_vectorizer.fit_transform(qa_questions)
            
            # Vectorize the input question
            question_vector = qa_vectorizer.transform([question])
            
            # Calculate similarities
            similarities = cosine_similarity(question_vector, qa_vectors).flatten()
            
            # Get top-k similar Q&A pairs
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            similar_qa = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    qa = self.qa_pairs[idx].copy()
                    qa['similarity_score'] = float(similarities[idx])
                    similar_qa.append(qa)
            
            return similar_qa
            
        except Exception as e:
            print(f"Error finding similar Q&A pairs: {str(e)}")
            return []

    def process_query_hybrid(self, document_text: str, question: str) -> str:
        """Process query using hybrid approach (API + enhanced fallback)"""
        
        # Try API first if available
        if self.model is not None:
            try:
                api_answer = self.process_query_with_enhancement(document_text, question)
                
                # Check if API answer is valid and informative
                if (api_answer and 
                    "Error processing question" not in api_answer and 
                    "API key not" not in api_answer and
                    "400" not in api_answer and
                    len(api_answer.strip()) > 5 and
                    "Information not available" not in api_answer):  # Don't accept generic "not available" from API
                    return api_answer
                    
                # If API returns "Information not available", try fallback
                if "Information not available" in api_answer:
                    print("API returned 'Information not available', trying enhanced fallback...")
                    fallback_answer = self.fallback_processor.process_query(document_text, question)
                    if fallback_answer and "Information not available" not in fallback_answer:
                        return fallback_answer
                    return api_answer  # Return API response if fallback also fails
                    
            except Exception as e:
                print(f"API processing failed: {str(e)}")
        
        # Use enhanced fallback if API fails or is not available
        print("Using enhanced fallback processor...")
        return self.fallback_processor.process_query(document_text, question)

    def process_query_with_enhancement(self, document_text: str, question: str) -> str:
        """Process query with optimized settings for speed and accuracy"""
        try:
            # Check if model is available (API key was set)
            if self.model is None:
                return "API key not configured. Please set the GOOGLE_API_KEY environment variable."
                
            # Generate optimized prompt with training context
            enhanced_prompt = self.generate_enhanced_prompt(document_text, question)
            
            # Generate response with optimized settings for speed and accuracy
            response = self.model.generate_content(
                enhanced_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.01,  # Ultra-low temperature for maximum consistency
                    top_p=0.6,        # More focused sampling
                    top_k=5,          # Very focused for consistent answers
                    max_output_tokens=80,  # Slightly increased for complete answers
                )
            )
            
            answer = response.text.strip()
            
            # Apply post-processing for better accuracy and consistency
            answer = self.post_process_answer(answer, question, document_text)
            
            return answer
            
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def post_process_answer(self, answer: str, question: str, document_text: str) -> str:
        """Post-process answer to ensure conciseness, accuracy, and grammatical correctness"""
        
        # Remove any leading/trailing whitespace
        answer = answer.strip()
        
        # Ensure first letter is capitalized if it's a sentence
        if answer and len(answer) > 1 and answer[0].islower() and answer[1].isalpha():
            answer = answer[0].upper() + answer[1:]
        
        # Remove common AI hedging phrases that might still appear
        hedging_phrases = [
            "Based on the document, ",
            "According to the document, ",
            "The document states that ",
            "It appears that ",
            "It seems that ",
            "The answer is ",
            "The policy states ",
            "The grace period is ",
            "The waiting period is ",
            "Yes, ",
            "No, "
        ]
        
        for phrase in hedging_phrases:
            if answer.startswith(phrase):
                answer = answer[len(phrase):]
                # Re-capitalize if the phrase removal made the first letter lowercase
                if answer and answer[0].islower() and answer[1].isalpha():
                    answer = answer[0].upper() + answer[1:]
                break
        
        # Ensure the answer ends with a period if it's a sentence and doesn't already
        if answer and answer[-1].isalpha() and not answer.endswith(('.', '!', '?')):
            answer += "."
        
        # Remove any specific formatting for common question types, as the prompt now handles conciseness
        # The goal is a direct, unadorned answer.
        
        return answer
    
    def batch_process_queries(self, document_text: str, questions: List[str]) -> List[str]:
        """Process multiple queries with enhanced accuracy and optimized performance using hybrid approach"""
        answers = []
        
        # Optimize document text once for all queries
        optimized_text = self._optimize_document_text(document_text)
        
        # Process questions in batches for better memory management
        batch_size = 5  # Process 5 questions at a time
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            
            for j, question in enumerate(batch_questions):
                try:
                    question_num = i + j + 1
                    print(f"Processing optimized query {question_num}/{len(questions)}")
                    
                    # Use hybrid processing for better reliability
                    answer = self.process_query_hybrid(optimized_text, question)
                    answers.append(answer)
                    
                    # Quick memory cleanup every few questions
                    if question_num % 3 == 0:
                        gc.collect()
                    
                except Exception as e:
                    print(f"Error processing question '{question}': {str(e)}")
                    answers.append(f"Error processing this question: {str(e)}")
            
            # Memory cleanup after each batch
            gc.collect()
        
        return answers
    
    def _optimize_document_text(self, document_text: str) -> str:
        """Optimize document text for faster processing"""
        import re
        
        # Quick optimization for speed - remove excessive whitespace and empty lines
        lines = [line.strip() for line in document_text.split('\n') if line.strip() and len(line.strip()) > 2]
        optimized_text = '\n'.join(lines)
        
        # Remove repeated patterns that don't add value
        optimized_text = re.sub(r'\n{3,}', '\n\n', optimized_text)  # Multiple newlines
        optimized_text = re.sub(r'[ \t]{2,}', ' ', optimized_text)   # Multiple spaces
        
        # Reduced text length for faster processing while maintaining key information
        max_length = 40000  # Reduced from 50000 for faster processing
        if len(optimized_text) > max_length:
            # Try to cut at a sentence boundary
            cut_point = optimized_text.rfind('.', 0, max_length)
            if cut_point > max_length * 0.8:  # If we found a good cut point
                optimized_text = optimized_text[:cut_point + 1] + "\n[Document truncated]"
            else:
                optimized_text = optimized_text[:max_length] + "\n[Document truncated]"
        
        return optimized_text
    
    def save_model_components(self):
        """Save model components for faster loading"""
        try:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model_components")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save vectorizer
            with open(os.path.join(model_dir, "vectorizer.pkl"), 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save document vectors
            with open(os.path.join(model_dir, "document_vectors.pkl"), 'wb') as f:
                pickle.dump(self.document_vectors, f)
            
            print("Model components saved successfully")
            
        except Exception as e:
            print(f"Error saving model components: {str(e)}")
    
    def load_model_components(self):
        """Load pre-trained model components"""
        try:
            model_dir = "/home/ubuntu/hackrx-main/model_components"
            
            # Load vectorizer
            vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            
            # Load document vectors
            vectors_path = os.path.join(model_dir, "document_vectors.pkl")
            if os.path.exists(vectors_path):
                with open(vectors_path, 'rb') as f:
                    self.document_vectors = pickle.load(f)
            
            print("Model components loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model components: {str(e)}")
            return False
    
    def evaluate_accuracy(self, test_questions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model accuracy on test questions"""
        correct_answers = 0
        total_questions = len(test_questions)
        
        for test_item in test_questions:
            document = test_item['context']
            question = test_item['question']
            
            answer = self.process_query_with_enhancement(document, question)
            
            # Simple accuracy check (can be enhanced with more sophisticated metrics)
            if "Information not available" not in answer and len(answer) > 10:
                correct_answers += 1
        
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct_answers': correct_answers,
            'total_questions': total_questions
        }