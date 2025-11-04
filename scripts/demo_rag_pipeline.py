#!/usr/bin/env python3
"""
Demo script for the Mongolian History RAG pipeline.

This script demonstrates the pipeline functionality without requiring
an actual OpenAI API key by using mock embeddings.
"""

import json
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any
import faiss


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class MockEmbeddingGenerator:
    """Mock embedding generator for demo purposes."""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        np.random.seed(42)  # For reproducible results
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings based on text characteristics."""
        embeddings = []
        
        for text in texts:
            # Create embeddings based on text characteristics
            # This is a simplified approach for demo purposes
            
            # Base embedding
            embedding = np.random.normal(0, 0.1, self.dimension)
            
            # Add features based on text content
            text_lower = text.lower()
            
            # Historical period features
            if 'xiii зуун' in text_lower or 'чингис' in text_lower:
                embedding[0:100] += 0.5  # 13th century cluster
            elif 'xx зуун' in text_lower or '1921' in text_lower:
                embedding[100:200] += 0.5  # 20th century cluster
            elif 'xvii' in text_lower or 'богд хаан' in text_lower:
                embedding[200:300] += 0.5  # 17th-19th century cluster
            
            # Content type features
            if 'нууц товчоо' in text_lower:
                embedding[300:400] += 0.3  # Secret History
            elif 'wikipedia' in text_lower:
                embedding[400:500] += 0.3  # Wikipedia
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)


def create_demo_embeddings():
    """Create demo embeddings and FAISS index."""
    logger = logging.getLogger(__name__)
    
    # Load dataset
    dataset_path = Path('data/mongolian_history_unified.jsonl')
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return False
    
    logger.info("Loading dataset...")
    entries = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    
    logger.info(f"Loaded {len(entries)} entries")
    
    # Create chunks and metadata
    logger.info("Creating chunks...")
    chunks = []
    metadata = []
    
    for entry_id, entry in enumerate(entries):
        text = entry.get('text', '')
        if not text:
            continue
        
        # Simple chunking (split by paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for chunk_id, paragraph in enumerate(paragraphs):
            if len(paragraph) < 100:  # Skip very short paragraphs
                continue
            
            # Limit chunk size
            if len(paragraph) > 1000:
                paragraph = paragraph[:1000]
            
            chunks.append(paragraph)
            metadata.append({
                'entry_id': entry_id,
                'chunk_id': chunk_id,
                'text': paragraph,
                'title': entry.get('title', ''),
                'source': entry.get('source', ''),
                'period': entry.get('period', ''),
                'date': entry.get('date', ''),
                'content_length': len(paragraph)
            })
    
    logger.info(f"Created {len(chunks)} chunks")
    
    # Generate mock embeddings
    logger.info("Generating mock embeddings...")
    embedder = MockEmbeddingGenerator()
    embeddings = embedder.generate_embeddings(chunks)
    
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    
    # Create FAISS index
    logger.info("Building FAISS index...")
    dimension = embeddings.shape[1]
    
    # Use simple flat index for demo
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    logger.info(f"Index built with {index.ntotal} vectors")
    
    # Save everything
    output_dir = Path('data/embeddings_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save index
    faiss.write_index(index, str(output_dir / 'mongolian_history_demo.faiss'))
    
    # Save metadata
    with open(output_dir / 'metadata_demo.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save config
    config = {
        'embedding_model': 'mock-embedding-model',
        'embedding_dimension': dimension,
        'chunk_size': 1000,
        'index_type': 'Flat',
        'total_chunks': len(metadata),
        'is_demo': True
    }
    
    with open(output_dir / 'config_demo.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Demo pipeline saved to {output_dir}")
    return True


def test_demo_retrieval():
    """Test the demo retrieval system."""
    logger = logging.getLogger(__name__)
    
    embeddings_dir = Path('data/embeddings_demo')
    
    # Load components
    logger.info("Loading demo pipeline...")
    
    # Load index
    index = faiss.read_index(str(embeddings_dir / 'mongolian_history_demo.faiss'))
    
    # Load metadata
    with open(embeddings_dir / 'metadata_demo.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load config
    with open(embeddings_dir / 'config_demo.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    logger.info(f"Loaded index with {index.ntotal} vectors")
    
    # Test search with mock query embedding
    logger.info("Testing search...")
    
    # Create mock query embedding (simulate "Чингис хаан" query)
    embedder = MockEmbeddingGenerator(config['embedding_dimension'])
    query_embedding = embedder.generate_embeddings(["Чингис хаан хэзээ төрсөн бэ?"])
    
    # Search
    k = 5
    scores, indices = index.search(query_embedding, k)
    
    logger.info(f"Search results for 'Чингис хаан' query:")
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:
            continue
        
        result_metadata = metadata[idx]
        
        print(f"\n{i+1}. Score: {score:.4f}")
        print(f"   Period: {result_metadata.get('period', 'N/A')}")
        print(f"   Source: {result_metadata.get('source', 'N/A')}")
        print(f"   Text: {result_metadata['text'][:150]}...")
    
    return True


def analyze_demo_data():
    """Analyze the demo dataset."""
    logger = logging.getLogger(__name__)
    
    embeddings_dir = Path('data/embeddings_demo')
    
    if not embeddings_dir.exists():
        logger.error("Demo embeddings not found. Run create_demo_embeddings() first.")
        return False
    
    # Load metadata
    with open(embeddings_dir / 'metadata_demo.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Analyze data
    from collections import Counter
    
    periods = [m.get('period', 'Unknown') for m in metadata]
    sources = [m.get('source', 'Unknown') for m in metadata]
    content_lengths = [m.get('content_length', 0) for m in metadata]
    
    print("\nDemo Dataset Analysis:")
    print("=" * 40)
    
    print(f"Total chunks: {len(metadata)}")
    print(f"Total entries: {len(set(m['entry_id'] for m in metadata))}")
    
    print(f"\nPeriod distribution:")
    period_counts = Counter(periods)
    for period, count in period_counts.most_common():
        print(f"  {period}: {count} chunks")
    
    print(f"\nSource distribution:")
    source_counts = Counter(sources)
    for source, count in source_counts.most_common():
        print(f"  {source}: {count} chunks")
    
    print(f"\nContent statistics:")
    print(f"  Average chunk length: {np.mean(content_lengths):.0f} characters")
    print(f"  Min chunk length: {min(content_lengths)} characters")
    print(f"  Max chunk length: {max(content_lengths)} characters")
    
    return True


def main():
    """Main demo function."""
    logger = setup_logging()
    
    print("Mongolian History RAG Pipeline Demo")
    print("=" * 40)
    print("This demo uses mock embeddings to demonstrate the pipeline")
    print("without requiring an OpenAI API key.\n")
    
    # Check if dataset exists
    dataset_path = Path('data/mongolian_history_unified.jsonl')
    if not dataset_path.exists():
        print("❌ Dataset not found. Please run the data cleaning script first:")
        print("   python scripts/clean_and_merge_json.py")
        return False
    
    # Create demo embeddings
    print("1. Creating demo embeddings...")
    if not create_demo_embeddings():
        print("❌ Failed to create demo embeddings")
        return False
    print("✅ Demo embeddings created")
    
    # Test retrieval
    print("\n2. Testing demo retrieval...")
    if not test_demo_retrieval():
        print("❌ Failed to test retrieval")
        return False
    print("✅ Demo retrieval working")
    
    # Analyze data
    print("\n3. Analyzing demo data...")
    analyze_demo_data()
    
    print("\n" + "=" * 40)
    print("🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Run: python scripts/setup_rag_pipeline.py")
    print("3. Use: python -m mongolian_rag.rag_agent")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)