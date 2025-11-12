#!/usr/bin/env python3
"""
Quick RAG Demo - Generate human-like answers with GPT
Simple script to test RAG with your dataset
"""

import json
import os
from pathlib import Path
from getpass import getpass


def search_documents(query: str, dataset_path: str, top_k: int = 3):
    """Simple search for relevant documents."""
    documents = []
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    
    # Simple scoring
    query_lower = query.lower()
    scored = []
    
    for doc in documents:
        text = doc.get('text', '').lower()
        score = 0
        
        # Check if query words are in text
        for word in query_lower.split():
            if word in text:
                score += text.count(word)
        
        if score > 0:
            scored.append((score, doc))
    
    # Sort and return top results
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_k]]


def generate_answer(question: str, api_key: str, model: str = "gpt-4o-mini"):
    """Generate answer using GPT with context from dataset."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    # Search for context
    print("🔍 Searching dataset...")
    docs = search_documents(
        question,
        "data/mongolian_history_unified_filtered.jsonl",
        top_k=3
    )
    
    if not docs:
        return "Уучлаарай, энэ асуултад хариулах мэдээлэл олдсонгүй."
    
    # Prepare context
    context = "\n\n".join([
        f"[Эх сурвалж {i+1}]: {doc.get('text', '')[:600]}"
        for i, doc in enumerate(docs)
    ])
    
    print(f"✅ Found {len(docs)} relevant sources")
    print("🤖 Generating answer...")
    
    # Generate answer
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """Та Монголын түүхийн мэргэжилтэн юм. 
Өгөгдсөн эх сурвалжид үндэслэн асуултад монгол хэлээр, 
байгалийн хүнлэг яриагаар хариулна уу."""
            },
            {
                "role": "user",
                "content": f"""Эх сурвалж:

{context}

Асуулт: {question}

Хариулт:"""
            }
        ],
        temperature=0.7,
        max_tokens=400
    )
    
    answer = response.choices[0].message.content.strip()
    
    # Show sources
    print("\n" + "=" * 60)
    print("💬 ANSWER:")
    print("=" * 60)
    print(answer)
    print("\n" + "=" * 60)
    print("📚 SOURCES:")
    print("=" * 60)
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.get('source', 'Unknown')} - {doc.get('period', '')}")
    print("=" * 60)
    
    return answer


def main():
    """Quick demo."""
    print("🇲🇳 Quick RAG Demo with GPT")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = getpass("Enter OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ API key required")
        return
    
    # Test questions
    questions = [
        "Чингис хаан хэзээ төрсөн бэ?",
        "Монголын ардчилсан хувьсгал хэзээ болсон бэ?",
        "Өгэдэй хааны тухай хэлнэ үү?"
    ]
    
    print("\n🧪 Testing with sample questions:\n")
    
    for question in questions:
        print(f"\n❓ {question}")
        print("-" * 60)
        
        try:
            generate_answer(question, api_key)
            print("\n")
            input("Press Enter for next question...")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Interactive
    print("\n💬 Interactive Mode (type 'exit' to quit)")
    print("=" * 60)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if not question or question.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break
        
        try:
            generate_answer(question, api_key)
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
