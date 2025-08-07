import os
import numpy as np
from typing import List
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.memory_manager import MemoryManager
from src.training.enhanced_fallback_processor import EnhancedFallbackProcessor


class EnhancedQueryProcessor:
    """
    Improved query processor using Gemini + Sentence Embeddings
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-1.5-pro-latest")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.fallback_processor = EnhancedFallbackProcessor()

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        return self.embedder.encode(chunks, show_progress_bar=False)

    def retrieve_relevant_chunks(self, question: str, chunks: List[str], top_k: int = 3) -> List[str]:
        question_embedding = self.embedder.encode([question])[0]
        chunk_embeddings = self.embed_chunks(chunks)

        scores = cosine_similarity([question_embedding], chunk_embeddings)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        return [chunks[i] for i in top_indices]

    def build_prompt(self, question: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)
        return f"""You are an assistant answering questions based on insurance policy documents.

Answer the following question using only the context below. If not found, respond with "Not mentioned in the document".

Context:
{context}

Question: {question}
"""

    @MemoryManager.cleanup_decorator
    def answer_question(self, question: str, document_chunks: List[str]) -> str:
        try:
            relevant_chunks = self.retrieve_relevant_chunks(question, document_chunks)
            prompt = self.build_prompt(question, relevant_chunks)

            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            return answer or "No answer found"
        except Exception as e:
            print(f"[Gemini Error] {e}")
            return self.fallback_processor.answer(question, document_chunks)

    def batch_answer(self, questions: List[str], document_chunks: List[str]) -> List[str]:
        return [self.answer_question(q, document_chunks) for q in questions]
