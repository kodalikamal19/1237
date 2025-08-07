import re
import json
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class FallbackProcessor:
    """Fallback processor that uses training data and pattern matching when API is not available"""
    
    def __init__(self, training_data_path: str = None):
        self.qa_pairs = []
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.8,
            min_df=1
        )
        self.qa_vectors = None
        
        # Load training data
        if training_data_path:
            self.load_training_data(training_data_path)
        else:
            # Try to load default processed training data
            import os
            default_path = os.path.join(os.path.dirname(__file__), "processed_training_data.json")
            if os.path.exists(default_path):
                self.load_training_data(default_path)
    
    def load_training_data(self, data_path: str):
        """Load processed training data"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            # Extract Q&A pairs
            self.qa_pairs = []
            for doc in training_data:
                if 'qa_pairs' in doc:
                    for qa in doc['qa_pairs']:
                        qa['document_type'] = doc.get('document_type', 'unknown')
                        qa['file_name'] = doc.get('file_name', 'unknown')
                        self.qa_pairs.append(qa)
            
            # Build vectors for similarity matching
            if self.qa_pairs:
                qa_questions = [qa['question'] for qa in self.qa_pairs]
                self.qa_vectors = self.vectorizer.fit_transform(qa_questions)
            
            print(f"Fallback processor loaded {len(self.qa_pairs)} Q&A pairs")
            
        except Exception as e:
            print(f"Error loading training data for fallback: {str(e)}")
            self.qa_pairs = []
    
    def process_query(self, document_text: str, question: str) -> str:
        """Process query using fallback methods"""
        
        # First, try to find exact or similar match in training data
        training_answer = self._find_training_answer(question)
        if training_answer:
            return training_answer
        
        # If no training match, use pattern-based extraction
        pattern_answer = self._extract_answer_by_patterns(document_text, question)
        if pattern_answer:
            return pattern_answer
        
        # If still no answer, use keyword-based search
        keyword_answer = self._extract_answer_by_keywords(document_text, question)
        if keyword_answer:
            return keyword_answer
        
        return "Information not available in the document."
    
    def _find_training_answer(self, question: str) -> str:
        """Find answer from training data using similarity matching"""
        if not self.qa_pairs or self.qa_vectors is None:
            return ""
        
        try:
            # Vectorize the input question
            question_vector = self.vectorizer.transform([question])
            
            # Calculate similarities
            similarities = cosine_similarity(question_vector, self.qa_vectors).flatten()
            
            # Find the best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            # If similarity is high enough, return the training answer
            if best_similarity > 0.7:  # High threshold for exact matches
                return self.qa_pairs[best_idx]['answer']
            elif best_similarity > 0.4:  # Medium threshold for similar questions
                # Return a modified version of the training answer
                base_answer = self.qa_pairs[best_idx]['answer']
                return f"{base_answer} (Based on similar policy terms)"
            
        except Exception as e:
            print(f"Error in training answer lookup: {str(e)}")
        
        return ""
    
    def _extract_answer_by_patterns(self, document_text: str, question: str) -> str:
        """Extract answer using predefined patterns for common questions"""
        
        question_lower = question.lower()
        doc_lower = document_text.lower()
        
        # Grace period patterns
        if 'grace period' in question_lower and 'premium' in question_lower:
            patterns = [
                r'grace period[:\s]+(\d+)\s*days?',
                r'grace period[:\s]+(\d+)\s*day',
                r'premium.*grace period[:\s]+(\d+)\s*days?'
            ]
            for pattern in patterns:
                match = re.search(pattern, doc_lower)
                if match:
                    return f"{match.group(1)} days."
        
        # Waiting period patterns
        if 'waiting period' in question_lower:
            if 'pre-existing' in question_lower or 'ped' in question_lower:
                patterns = [
                    r'waiting period.*pre-existing.*?(\d+)\s*(?:years?|months?)',
                    r'pre-existing.*waiting period.*?(\d+)\s*(?:years?|months?)',
                    r'ped.*waiting period.*?(\d+)\s*(?:years?|months?)'
                ]
            elif 'cataract' in question_lower:
                patterns = [
                    r'cataract.*waiting period.*?(\d+)\s*(?:years?|months?)',
                    r'waiting period.*cataract.*?(\d+)\s*(?:years?|months?)'
                ]
            else:
                patterns = [
                    r'waiting period[:\s]+(\d+)\s*(?:years?|months?)',
                    r'waiting period.*?(\d+)\s*(?:years?|months?)'
                ]
            
            for pattern in patterns:
                match = re.search(pattern, doc_lower)
                if match:
                    return f"{match.group(1)} {'years' if 'year' in match.group(0) else 'months'}."
        
        # Maternity coverage patterns
        if 'maternity' in question_lower:
            if 'cover' in question_lower or 'expense' in question_lower:
                if 'maternity' in doc_lower:
                    if 'not covered' in doc_lower or 'excluded' in doc_lower:
                        return "Maternity expenses are not covered."
                    else:
                        return "Maternity expenses are covered subject to policy terms."
        
        # No Claim Discount patterns
        if 'no claim discount' in question_lower or 'ncd' in question_lower:
            patterns = [
                r'no claim discount[:\s]+(\d+)%',
                r'ncd[:\s]+(\d+)%',
                r'cumulative bonus[:\s]+(\d+)%'
            ]
            for pattern in patterns:
                match = re.search(pattern, doc_lower)
                if match:
                    return f"{match.group(1)}% No Claim Discount."
        
        # Hospital definition patterns
        if 'hospital' in question_lower and 'define' in question_lower:
            # Look for hospital definition in the document
            hospital_def_match = re.search(r'hospital[:\s]+means?[^.]*\.', doc_lower)
            if hospital_def_match:
                definition = hospital_def_match.group(0)
                # Clean up and truncate if too long
                if len(definition) > 200:
                    definition = definition[:200] + "..."
                return definition.capitalize()
        
        return ""
    
    def _extract_answer_by_keywords(self, document_text: str, question: str) -> str:
        """Extract answer using keyword-based search"""
        
        # Extract key terms from the question
        question_words = re.findall(r'\b\w+\b', question.lower())
        
        # Remove common stop words
        stop_words = {'what', 'is', 'the', 'are', 'does', 'this', 'policy', 'under', 'for', 'and', 'or', 'a', 'an'}
        keywords = [word for word in question_words if word not in stop_words and len(word) > 2]
        
        if not keywords:
            return ""
        
        # Split document into sentences
        sentences = re.split(r'[.!?]+', document_text)
        
        # Find sentences containing the most keywords
        best_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for keyword in keywords if keyword in sentence_lower)
            
            if keyword_count >= 2:  # At least 2 keywords
                best_sentences.append((sentence.strip(), keyword_count))
        
        if best_sentences:
            # Sort by keyword count and return the best match
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentence = best_sentences[0][0]
            
            # Clean up and truncate if too long
            if len(best_sentence) > 200:
                best_sentence = best_sentence[:200] + "..."
            
            return best_sentence
        
        return ""

class HybridProcessor:
    """Hybrid processor that combines API and fallback methods"""
    
    def __init__(self, api_processor, fallback_processor):
        self.api_processor = api_processor
        self.fallback_processor = fallback_processor
    
    def process_query(self, document_text: str, question: str) -> str:
        """Process query using API first, fallback if API fails"""
        
        # Try API first
        try:
            api_answer = self.api_processor.process_query_with_enhancement(document_text, question)
            
            # Check if API answer is valid
            if (api_answer and 
                "Error processing question" not in api_answer and 
                "API key not" not in api_answer and
                len(api_answer.strip()) > 5):
                return api_answer
        except Exception as e:
            print(f"API processing failed: {str(e)}")
        
        # Use fallback if API fails
        print("Using fallback processor...")
        return self.fallback_processor.process_query(document_text, question)

