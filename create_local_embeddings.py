#!/usr/bin/env python3
"""
Create Local Embeddings without OpenAI API
Uses sentence-transformers for free local embeddings
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import pickle


def install_dependencies():
    """Install required packages for local embeddings."""
    import subprocess
    import sys
    
    print("📦 Installing sentence-transformers...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "sentence-transformers", "faiss-cpu"
        ])
        print("✅ Dependencies installed")
        return True
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_local_embeddings(
    dataset_path: str = "data/mongolian_history_unified_filtered.jsonl",
    output_dir: str = "data/embeddings_local"
):
    """Create embeddings using local sentence-transformers model."""
    
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        print("📦 Required packages not found. Installing...")
        if not install_dependencies():
            return False
        from sentence_transformers import SentenceTransformer
        import faiss
    
    print("\n🚀 Creating Local Embeddings (No API Key Needed)")
    print("=" * 60)
    
    # Load model (downloads on first use, ~400MB)
    print("\n📥 Loading embedding model...")
    print("   Model: paraphrase-multilingual-MiniLM-L12-v2")
    print("   Size: ~400MB (downloads once, cached locally)")
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✅ Model loaded")
    
    # Load dataset
    print(f"\n📂 Loading dataset: {dataset_path}")
    documents = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Create chunks
    print("\n✂️ Creating text chunks...")
    chunks = []
    metadata = []
    
    for doc_idx, doc in enumerate(documents):
        text = doc.get('text', '')
        
        # Split into chunks (500 chars with 50 char overlap)
        chunk_size = 500
        overlap = 50
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
            
            chunks.append(chunk_text)
            metadata.append({
                'doc_index': doc_idx,
                'chunk_index': len(chunks) - 1,
                'source': doc.get('source', ''),
                'title': doc.get('title', ''),
                'chapter': doc.get('chapter', ''),
                'period': doc.get('period', ''),
                'text': chunk_text
            })
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Generate embeddings
    print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
    print("   This may take 1-2 minutes...")
    
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        batch_size=32
    )
    
    print(f"✅ Generated embeddings: shape {embeddings.shape}")
    
    # Create FAISS index
    print("\n🗂️ Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"✅ FAISS index created with {index.ntotal} vectors")
    
    # Save everything
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {output_dir}...")
    
    # Save FAISS index
    faiss.write_index(index, str(output_path / "mongolian_history.faiss"))
    
    # Save metadata
    with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Save config
    config = {
        'model': 'paraphrase-multilingual-MiniLM-L12-v2',
        'dimension': dimension,
        'total_chunks': len(chunks),
        'total_documents': len(documents),
        'dataset': dataset_path,
        'embedding_type': 'local_sentence_transformers'
    }
    
    with open(output_path / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print("✅ All files saved")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Local embeddings created")
    print("=" * 60)
    print(f"\n📊 Statistics:")
    print(f"   • Documents: {len(documents)}")
    print(f"   • Chunks: {len(chunks)}")
    print(f"   • Embedding dimension: {dimension}")
    print(f"   • Index size: {index.ntotal} vectors")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   • mongolian_history.faiss - Vector index")
    print(f"   • metadata.json - Document metadata")
    print(f"   • config.json - Configuration")
    
    print("\n✅ No API key needed!")
    print("✅ No costs!")
    print("✅ Runs completely offline!")
    
    return True


def test_search(output_dir: str = "data/embeddings_local"):
    """Test the created embeddings with a sample search."""
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        print("❌ Required packages not installed")
        return
    
    print("\n🧪 Testing search functionality...")
    
    # Load index and metadata
    index = faiss.read_index(str(Path(output_dir) / "mongolian_history.faiss"))
    
    with open(Path(output_dir) / "metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load model
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Test query
    query = "Чингис хаан"
    print(f"\n❓ Test query: {query}")
    
    # Generate query embedding
    query_embedding = model.encode([query])[0]
    
    # Search
    k = 3
    distances, indices = index.search(
        query_embedding.reshape(1, -1).astype('float32'),
        k
    )
    
    print(f"\n📚 Top {k} results:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        meta = metadata[idx]
        print(f"\n{i}. Score: {1/(1+dist):.3f}")
        print(f"   Source: {meta['source']}")
        print(f"   Period: {meta['period']}")
        print(f"   Text: {meta['text'][:150]}...")


def main():
    """Main execution."""
    print("🇲🇳 Local Embedding Creator")
    print("No OpenAI API key required!")
    print("=" * 60)
    
    # Create embeddings
    success = create_local_embeddings()
    
    if success:
        # Test search
        test_search()
        
        print("\n" + "=" * 60)
        print("🚀 Ready to use!")
        print("=" * 60)
        print("\nYou can now use these embeddings with your RAG system.")
        print("The embeddings work offline and have no API costs.")
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
