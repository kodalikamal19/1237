"""
Dataset processor for extracting and preparing training data from PDF documents
"""
import os
import json
import pypdf
import io
from typing import List, Dict, Any
from src.utils.memory_manager import MemoryManager
import gc

class DatasetProcessor:
    """Process PDF documents to create training dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.memory_manager = MemoryManager()
        self.training_data = []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a single PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text_parts = []
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text.strip())
                        
                        # Memory management
                        if page_num % 10 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        print(f"Warning: Failed to extract text from page {page_num + 1} in {pdf_path}: {str(e)}")
                        continue
                
                full_text = "\n\n".join(text_parts)
                return full_text
                
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return ""
    
    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """Process all PDFs in the dataset directory"""
        dataset = []
        
        if not os.path.exists(self.dataset_path):
            print(f"Dataset path {self.dataset_path} does not exist")
            return dataset
        
        pdf_files = [f for f in os.listdir(self.dataset_path) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for i, pdf_file in enumerate(pdf_files):
            print(f"Processing {i+1}/{len(pdf_files)}: {pdf_file}")
            
            pdf_path = os.path.join(self.dataset_path, pdf_file)
            text_content = self.extract_text_from_pdf(pdf_path)
            
            if text_content:
                # Create training entry
                entry = {
                    'filename': pdf_file,
                    'content': text_content,
                    'length': len(text_content),
                    'document_type': self._classify_document_type(pdf_file, text_content),
                    'key_sections': self._extract_key_sections(text_content)
                }
                dataset.append(entry)
                print(f"  - Extracted {len(text_content)} characters")
            else:
                print(f"  - Failed to extract text from {pdf_file}")
            
            # Memory cleanup
            gc.collect()
        
        return dataset
    
    def _classify_document_type(self, filename: str, content: str) -> str:
        """Classify document type based on filename and content"""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        if 'policy' in filename_lower or 'policy' in content_lower:
            return 'insurance_policy'
        elif 'health' in filename_lower or 'health' in content_lower:
            return 'health_insurance'
        elif 'life' in filename_lower or 'life' in content_lower:
            return 'life_insurance'
        elif 'general' in filename_lower or 'general' in content_lower:
            return 'general_insurance'
        else:
            return 'insurance_document'
    
    def _extract_key_sections(self, content: str) -> Dict[str, str]:
        """Extract key sections from document content"""
        sections = {}
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        # Common insurance document sections
        section_keywords = [
            'policy', 'coverage', 'benefits', 'exclusions', 'premium', 
            'terms', 'conditions', 'definitions', 'claims', 'procedure'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            line_lower = line.lower()
            is_section_header = False
            
            for keyword in section_keywords:
                if keyword in line_lower and len(line) < 100:  # Likely a header
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    current_section = line
                    current_content = []
                    is_section_header = True
                    break
            
            if not is_section_header and current_section:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def generate_training_questions(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate training questions and answers from the dataset"""
        training_pairs = []
        
        # Common insurance-related questions
        question_templates = [
            "What is the coverage amount for {}?",
            "What are the exclusions in this policy?",
            "What is the premium amount?",
            "What are the key benefits of this policy?",
            "What are the terms and conditions?",
            "How to file a claim?",
            "What is the policy period?",
            "Who is eligible for this policy?",
            "What documents are required?",
            "What is the waiting period?",
            "What are the renewal conditions?",
            "What is covered under this policy?",
            "What is the sum insured?",
            "What are the policy features?",
            "What is the claim settlement process?"
        ]
        
        for doc in dataset:
            content = doc['content']
            filename = doc['filename']
            doc_type = doc['document_type']
            
            # Generate questions specific to this document
            for template in question_templates:
                if '{}' in template:
                    question = template.format(doc_type.replace('_', ' '))
                else:
                    question = template
                
                training_pair = {
                    'document': filename,
                    'document_type': doc_type,
                    'question': question,
                    'context': content[:8000],  # Limit context length
                    'answer_available': True  # Will be determined by model
                }
                training_pairs.append(training_pair)
        
        return training_pairs
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str):
        """Save processed dataset to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"Dataset saved to {output_path}")
        except Exception as e:
            print(f"Error saving dataset: {str(e)}")
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON file"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            print(f"Dataset loaded from {dataset_path}")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return []

