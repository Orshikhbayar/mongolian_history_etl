#!/usr/bin/env python3
"""
Simple RAG Demo - Works without embeddings
Uses basic text search instead of vector embeddings
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from difflib import SequenceMatcher


class SimpleMongoliannRAG:
    """Simple RAG system using text matching instead of embeddings."""
    
    def __init__(self, dataset_path: str = "data/mongolian_history_unified_filtered.jsonl"):
        self.dataset_path = Path(dataset_path)
        self.documents = []
        self.load_dataset()
    
    def load_dataset(self):
        """Load the unified dataset."""
        print(f"📂 Loading dataset from {self.dataset_path}...")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                self.documents.append(doc)
        
        print(f"✅ Loaded {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents using text similarity."""
        query_lower = query.lower()
        
        # Score each document
        scored_docs = []
        for doc in self.documents:
            text = doc.get('text', '').lower()
            title = doc.get('title', '').lower()
            
            # Calculate similarity score
            score = 0
            
            # Exact phrase match (highest score)
            if query_lower in text:
                score += 10
            
            # Title match
            if query_lower in title:
                score += 5
            
            # Word overlap
            query_words = set(query_lower.split())
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            score += overlap
            
            # Sequence similarity
            similarity = SequenceMatcher(None, query_lower, text[:500]).ratio()
            score += similarity * 2
            
            if score > 0:
                scored_docs.append({
                    'document': doc,
                    'score': score
                })
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return [item['document'] for item in scored_docs[:top_k]]
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using retrieved context."""
        print(f"\n❓ Question: {question}")
        print("=" * 60)
        
        # Retrieve relevant documents
        relevant_docs = self.search(question, top_k=3)
        
        if not relevant_docs:
            return {
                'question': question,
                'answer': 'Уучлаарай, энэ асуултад хариулах мэдээлэл олдсонгүй.',
                'sources': []
            }
        
        # Format context
        context_parts = []
        sources = []
        
        for i, doc in enumerate(relevant_docs, 1):
            text = doc.get('text', '')[:500]  # First 500 chars
            source = doc.get('source', 'Unknown')
            chapter = doc.get('chapter', '')
            period = doc.get('period', '')
            
            context_parts.append(f"[{i}] {text}...")
            sources.append({
                'source': source,
                'chapter': chapter,
                'period': period,
                'text_preview': text[:200]
            })
        
        # Create response
        response = {
            'question': question,
            'context': '\n\n'.join(context_parts),
            'sources': sources,
            'note': 'This is a simple text-based search. For better results, use the full RAG system with embeddings.'
        }
        
        # Display results
        print(f"\n📚 Found {len(relevant_docs)} relevant documents:\n")
        
        for i, source in enumerate(sources, 1):
            print(f"{i}. Source: {source['source']}")
            if source['chapter']:
                print(f"   Chapter: {source['chapter']}")
            if source['period']:
                print(f"   Period: {source['period']}")
            print(f"   Preview: {source['text_preview']}...\n")
        
        return response


def main():
    """Run simple RAG demo."""
    print("🇲🇳 Simple Mongolian History RAG Demo")
    print("=" * 60)
    print("This demo uses basic text search (no embeddings required)")
    print("=" * 60)
    
    # Initialize RAG
    rag = SimpleMongoliannRAG()
    
    # Test questions
    test_questions = [
        "Чингис хааны тухай",
        "Монголын нууц товчоо",
        "Тэмүжин",
        "Бөртэ үжин",
        "Жамуха"
    ]
    
    print("\n🧪 Testing with sample questions:\n")
    
    for question in test_questions:
        result = rag.answer_question(question)
        print("\n" + "-" * 60 + "\n")
    
    # Interactive mode
    print("\n💬 Interactive Mode (type 'exit' to quit)")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            result = rag.answer_question(question)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
