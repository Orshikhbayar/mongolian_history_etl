#!/usr/bin/env python3
"""
RAG System with GPT-4 for Human-like Answers
Retrieves context from dataset and generates natural responses using OpenAI
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from getpass import getpass


class MongolianRAGWithGPT:
    """RAG system that generates human-like answers using OpenAI."""
    
    def __init__(
        self,
        dataset_path: str = "data/mongolian_history_unified_filtered.jsonl",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        self.dataset_path = Path(dataset_path)
        self.documents = []
        self.model = model
        
        # Setup OpenAI
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key parameter.")
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.load_dataset()
    
    def load_dataset(self):
        """Load the filtered dataset."""
        print(f"📂 Loading dataset from {self.dataset_path}...")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                self.documents.append(doc)
        
        print(f"✅ Loaded {len(self.documents)} documents")
    
    def search_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents using text similarity."""
        query_lower = query.lower()
        
        # Score each document
        scored_docs = []
        for doc in self.documents:
            text = doc.get('text', '').lower()
            title = doc.get('title', '').lower()
            
            # Calculate similarity score
            score = 0
            
            # Exact phrase match
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
    
    def generate_answer(
        self,
        question: str,
        language: str = "mongolian",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate a human-like answer using GPT with retrieved context."""
        
        print(f"\n❓ Question: {question}")
        print("=" * 60)
        
        # Retrieve relevant context
        print("🔍 Searching for relevant context...")
        relevant_docs = self.search_context(question, top_k=3)
        
        if not relevant_docs:
            return {
                'question': question,
                'answer': 'Уучлаарай, энэ асуултад хариулах мэдээлэл олдсонгүй.',
                'sources': [],
                'context_used': False
            }
        
        # Prepare context
        context_parts = []
        sources = []
        
        for i, doc in enumerate(relevant_docs, 1):
            text = doc.get('text', '')
            source = doc.get('source', 'Unknown')
            chapter = doc.get('chapter', '')
            period = doc.get('period', '')
            title = doc.get('title', '')
            
            context_parts.append(f"[Эх сурвалж {i}]\n{text[:800]}")
            sources.append({
                'source': source,
                'title': title,
                'chapter': chapter,
                'period': period
            })
        
        context = "\n\n".join(context_parts)
        
        print(f"✅ Found {len(relevant_docs)} relevant sources")
        print("🤖 Generating answer with GPT...")
        
        # Create prompt based on language
        if language.lower() == "mongolian":
            system_prompt = """Та Монголын түүхийн мэргэжилтэн юм. Өгөгдсөн эх сурвалжийн мэдээлэлд үндэслэн асуултад хариулна уу.

Дүрэм:
1. Зөвхөн өгөгдсөн эх сурвалжийн мэдээлэлийг ашиглана
2. Монгол хэлээр тодорхой, ойлгомжтой хариулна
3. Мэдэхгүй бол "Өгөгдсөн эх сурвалжид энэ тухай мэдээлэл байхгүй байна" гэж хэлнэ
4. Хариултаа байгалийн, хүнлэг яриагаар өгнө
5. Эх сурвалжийн мэдээллийг нэмж тайлбарлаж болно"""
            
            user_prompt = f"""Эх сурвалжийн мэдээлэл:

{context}

Асуулт: {question}

Дээрх эх сурвалжид үндэслэн асуултад монгол хэлээр хариулна уу:"""
        
        else:  # English
            system_prompt = """You are a Mongolian history expert. Answer questions based on the provided source materials.

Rules:
1. Only use information from the provided sources
2. Answer clearly and naturally
3. If you don't know, say "The provided sources don't contain this information"
4. Write in a conversational, human-like tone
5. You can elaborate on the source information"""
            
            user_prompt = f"""Source materials:

{context}

Question: {question}

Based on the sources above, please answer the question:"""
        
        # Generate answer with GPT
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Display results
            print("\n" + "=" * 60)
            print("💬 ANSWER:")
            print("=" * 60)
            print(answer)
            print("\n" + "=" * 60)
            print("📚 SOURCES:")
            print("=" * 60)
            
            for i, source in enumerate(sources, 1):
                print(f"\n{i}. {source['source']}")
                if source['title']:
                    print(f"   Title: {source['title']}")
                if source['period']:
                    print(f"   Period: {source['period']}")
                if source['chapter']:
                    print(f"   Chapter: {source['chapter']}")
            
            return {
                'question': question,
                'answer': answer,
                'sources': sources,
                'context_used': True,
                'model': self.model
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                'question': question,
                'answer': f'Error: {str(e)}',
                'sources': sources,
                'context_used': False
            }


def main():
    """Interactive RAG with GPT."""
    print("🇲🇳 Mongolian History RAG with GPT")
    print("=" * 60)
    print("Generates human-like answers using OpenAI GPT")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("\n🔑 OpenAI API key not found in environment")
        api_key = getpass("Enter your OpenAI API key: ").strip()
        
        if not api_key:
            print("❌ API key required")
            return
    
    # Test API key
    print("\n🧪 Testing API key...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        client.models.list()
        print("✅ API key valid")
    except Exception as e:
        print(f"❌ Invalid API key: {e}")
        return
    
    # Choose model
    print("\n🤖 Choose model:")
    print("1. gpt-4o-mini (faster, cheaper)")
    print("2. gpt-4o (better quality)")
    print("3. gpt-4-turbo (balanced)")
    
    choice = input("\nChoice (1-3, default=1): ").strip() or "1"
    
    models = {
        "1": "gpt-4o-mini",
        "2": "gpt-4o",
        "3": "gpt-4-turbo"
    }
    
    model = models.get(choice, "gpt-4o-mini")
    print(f"✅ Using model: {model}")
    
    # Initialize RAG
    try:
        rag = MongolianRAGWithGPT(api_key=api_key, model=model)
    except Exception as e:
        print(f"❌ Error initializing RAG: {e}")
        return
    
    # Test questions
    print("\n🧪 Testing with sample questions:\n")
    
    test_questions = [
        "Чингис хаан хэзээ төрсөн бэ?",
        "Монголын ардчилсан хувьсгал хэзээ болсон бэ?",
        "Өгэдэй хааны тухай хэлнэ үү?"
    ]
    
    for question in test_questions:
        result = rag.generate_answer(question)
        print("\n" + "-" * 60 + "\n")
        input("Press Enter to continue...")
    
    # Interactive mode
    print("\n💬 Interactive Mode")
    print("=" * 60)
    print("Ask questions in Mongolian or English")
    print("Type 'exit' to quit")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Detect language
            has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in question)
            language = "mongolian" if has_cyrillic else "english"
            
            result = rag.generate_answer(question, language=language)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
