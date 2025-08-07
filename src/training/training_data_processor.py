import json
import os
import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

class TrainingDataProcessor:
    """Process extracted PDF data to create training dataset for the model"""
    
    def __init__(self, training_data_path: str):
        self.training_data_path = training_data_path
        self.processed_data = []
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.8,
            min_df=1
        )
        self.document_vectors = None
        
    def load_and_process_data(self):
        """Load and process the extracted PDF data"""
        try:
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            print(f"Loaded {len(raw_data)} documents")
            
            for doc in raw_data:
                processed_doc = self._process_document(doc)
                if processed_doc:
                    self.processed_data.append(processed_doc)
            
            print(f"Processed {len(self.processed_data)} documents")
            
            # Build document vectors for similarity search
            self._build_document_vectors()
            
            return True
            
        except Exception as e:
            print(f"Error loading training data: {str(e)}")
            return False
    
    def _process_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual document to extract key information"""
        try:
            content = doc['content']
            file_name = doc['file_name']
            
            # Clean and normalize text
            cleaned_content = self._clean_text(content)
            
            # Extract key sections
            sections = self._extract_sections(cleaned_content)
            
            # Generate question-answer pairs
            qa_pairs = self._generate_qa_pairs(sections, file_name)
            
            processed_doc = {
                'file_name': file_name,
                'content': cleaned_content,
                'sections': sections,
                'qa_pairs': qa_pairs,
                'document_type': self._identify_document_type(file_name, content)
            }
            
            return processed_doc
            
        except Exception as e:
            print(f"Error processing document {doc.get('file_name', 'unknown')}: {str(e)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'[^\w\s.,:;!?()-[\]{}"\'/\\@#$%^&*+=<>~`|]', '', text)
        
        # Normalize common insurance terms
        replacements = {
            'Sum Insured': 'sum insured',
            'Policy Period': 'policy period',
            'Grace Period': 'grace period',
            'Waiting Period': 'waiting period',
            'Pre-existing': 'pre-existing',
            'Co-payment': 'co-payment',
            'Deductible': 'deductible'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract key sections from document content"""
        sections = {}
        
        # Common insurance policy sections
        section_patterns = {
            'definitions': r'(DEFINITIONS|STANDARD DEFINITIONS)(.*?)(?=SECTION|COVERAGE|BENEFITS|$)',
            'coverage': r'(COVERAGE|BENEFITS|SCOPE OF COVER)(.*?)(?=EXCLUSIONS|SECTION|$)',
            'exclusions': r'(EXCLUSIONS|NOT COVERED)(.*?)(?=SECTION|CONDITIONS|$)',
            'conditions': r'(CONDITIONS|TERMS AND CONDITIONS)(.*?)(?=SECTION|CLAIMS|$)',
            'claims': r'(CLAIMS|CLAIM PROCEDURE)(.*?)(?=SECTION|$)',
            'premium': r'(PREMIUM|PAYMENT)(.*?)(?=SECTION|$)',
            'grace_period': r'(GRACE PERIOD|GRACE)(.*?)(?=\d+\.|SECTION|$)',
            'waiting_period': r'(WAITING PERIOD|WAITING)(.*?)(?=\d+\.|SECTION|$)'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(2).strip()[:2000]  # Limit section length
        
        return sections
    
    def _generate_qa_pairs(self, sections: Dict[str, str], file_name: str) -> List[Dict[str, str]]:
        """Generate question-answer pairs from document sections"""
        qa_pairs = []
        
        # Common insurance questions and their likely answers
        common_questions = [
            {
                'question': 'What is the grace period for premium payment?',
                'section': 'grace_period',
                'keywords': ['grace period', 'premium payment', 'days']
            },
            {
                'question': 'What is the waiting period for pre-existing diseases?',
                'section': 'waiting_period',
                'keywords': ['waiting period', 'pre-existing', 'diseases', 'months', 'years']
            },
            {
                'question': 'Does this policy cover maternity expenses?',
                'section': 'coverage',
                'keywords': ['maternity', 'pregnancy', 'childbirth', 'delivery']
            },
            {
                'question': 'What is the waiting period for cataract surgery?',
                'section': 'waiting_period',
                'keywords': ['cataract', 'surgery', 'waiting period', 'months']
            },
            {
                'question': 'Are medical expenses for organ donor covered?',
                'section': 'coverage',
                'keywords': ['organ donor', 'transplant', 'medical expenses']
            },
            {
                'question': 'What is the No Claim Discount offered?',
                'section': 'premium',
                'keywords': ['no claim discount', 'NCD', 'bonus', 'percentage']
            },
            {
                'question': 'Is there a benefit for preventive health check-ups?',
                'section': 'coverage',
                'keywords': ['preventive', 'health check', 'annual', 'screening']
            },
            {
                'question': 'How does the policy define a Hospital?',
                'section': 'definitions',
                'keywords': ['hospital', 'definition', 'beds', 'qualified']
            },
            {
                'question': 'What is the extent of coverage for AYUSH treatments?',
                'section': 'coverage',
                'keywords': ['AYUSH', 'treatment', 'coverage', 'alternative medicine']
            },
            {
                'question': 'Are there any sub-limits on room rent and ICU charges?',
                'section': 'coverage',
                'keywords': ['room rent', 'ICU', 'sub-limit', 'charges']
            }
        ]
        
        for q_template in common_questions:
            answer = self._find_answer_in_sections(q_template, sections)
            if answer:
                qa_pairs.append({
                    'question': q_template['question'],
                    'answer': answer,
                    'source_file': file_name
                })
        
        return qa_pairs
    
    def _find_answer_in_sections(self, q_template: Dict[str, Any], sections: Dict[str, str]) -> str:
        """Find answer for a question template in document sections"""
        # First, try the specific section
        target_section = q_template.get('section', '')
        if target_section in sections:
            section_text = sections[target_section]
            answer = self._extract_answer_from_text(section_text, q_template['keywords'])
            if answer:
                return answer
        
        # If not found, search all sections
        for section_name, section_text in sections.items():
            answer = self._extract_answer_from_text(section_text, q_template['keywords'])
            if answer:
                return answer
        
        return ""
    
    def _extract_answer_from_text(self, text: str, keywords: List[str]) -> str:
        """Extract answer from text based on keywords"""
        text_lower = text.lower()
        
        # Check if any keywords are present
        if not any(keyword.lower() in text_lower for keyword in keywords):
            return ""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Find sentences containing keywords
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword.lower() in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Return the first relevant sentence, cleaned up
            answer = relevant_sentences[0]
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            # Limit answer length
            if len(answer) > 200:
                answer = answer[:200] + "..."
            
            return answer
        
        return ""
    
    def _identify_document_type(self, file_name: str, content: str) -> str:
        """Identify the type of insurance document"""
        content_lower = content.lower()
        
        if 'health' in content_lower or 'medical' in content_lower:
            return 'health_insurance'
        elif 'life' in content_lower:
            return 'life_insurance'
        elif 'motor' in content_lower or 'vehicle' in content_lower:
            return 'motor_insurance'
        else:
            return 'general_insurance'
    
    def _build_document_vectors(self):
        """Build TF-IDF vectors for document similarity"""
        try:
            documents = [doc['content'] for doc in self.processed_data]
            self.document_vectors = self.vectorizer.fit_transform(documents)
            print(f"Built document vectors for {len(documents)} documents")
        except Exception as e:
            print(f"Error building document vectors: {str(e)}")
    
    def save_processed_data(self, output_path: str):
        """Save processed training data"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
            print(f"Saved processed data to {output_path}")
            
            # Save vectorizer and vectors
            model_dir = os.path.dirname(output_path)
            vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
            vectors_path = os.path.join(model_dir, "document_vectors.pkl")
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            with open(vectors_path, 'wb') as f:
                pickle.dump(self.document_vectors, f)
            
            print(f"Saved vectorizer and vectors to {model_dir}")
            
        except Exception as e:
            print(f"Error saving processed data: {str(e)}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about the training data"""
        total_docs = len(self.processed_data)
        total_qa_pairs = sum(len(doc['qa_pairs']) for doc in self.processed_data)
        
        doc_types = {}
        for doc in self.processed_data:
            doc_type = doc['document_type']
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        return {
            'total_documents': total_docs,
            'total_qa_pairs': total_qa_pairs,
            'document_types': doc_types,
            'average_qa_per_doc': total_qa_pairs / total_docs if total_docs > 0 else 0
        }

if __name__ == "__main__":
    # Process the extracted training data
    processor = TrainingDataProcessor("/home/ubuntu/extracted_pdf_text/extracted_training_data.json")
    
    if processor.load_and_process_data():
        # Save processed data
        output_path = "/home/ubuntu/hackrx-optimized/src/training/processed_training_data.json"
        processor.save_processed_data(output_path)
        
        # Print statistics
        stats = processor.get_training_statistics()
        print("\nTraining Data Statistics:")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Q&A Pairs: {stats['total_qa_pairs']}")
        print(f"Average Q&A per Document: {stats['average_qa_per_doc']:.2f}")
        print(f"Document Types: {stats['document_types']}")
    else:
        print("Failed to process training data")

