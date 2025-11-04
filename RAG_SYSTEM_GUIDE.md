# Mongolian History RAG System Guide

## Overview

This guide covers the complete Retrieval-Augmented Generation (RAG) system for Mongolian historical data. The system uses OpenAI's text-embedding-3-small model for embeddings and FAISS for efficient similarity search, enabling intelligent question-answering about Mongolian history.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│  Data Cleaning   │───▶│ Unified Dataset │
│   (JSON/JSONL)  │    │   & Merging      │    │    (JSONL)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RAG Agent     │◀───│ FAISS Retrieval │◀───│   Embeddings    │
│  (Q&A System)   │    │     Engine       │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components

### 1. Embedding Pipeline (`mongolian_rag/embedding_pipeline.py`)
- **Text Chunking**: Splits long texts into overlapping chunks
- **Embedding Generation**: Uses OpenAI's text-embedding-3-small model
- **FAISS Indexing**: Creates efficient similarity search index
- **Metadata Management**: Preserves source information and context

### 2. Retrieval Engine (`mongolian_rag/retrieval_engine.py`)
- **Similarity Search**: Finds relevant historical content
- **Result Ranking**: Scores and ranks search results
- **Filtering**: Search by period, source, or content type
- **Context Preparation**: Formats results for RAG applications

### 3. RAG Agent (`mongolian_rag/rag_agent.py`)
- **Question Answering**: Generates responses using retrieved context
- **Conversation Management**: Maintains chat history
- **Source Attribution**: Provides citations for answers
- **Multi-language Support**: Mongolian and English responses

## Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Cleaned Mongolian history dataset

### 1. Install Dependencies
```bash
pip install -r requirements_rag.txt
```

### 2. Set Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Run Setup Script
```bash
python scripts/setup_rag_pipeline.py
```

This script will:
- ✅ Check dependencies
- ✅ Verify API key
- ✅ Create embeddings from dataset
- ✅ Build FAISS index
- ✅ Test retrieval system
- ✅ Test RAG agent
- ✅ Create usage examples

## Usage Examples

### Basic Usage
```python
import os
from mongolian_rag import create_interactive_session

# Create RAG session
api_key = os.getenv('OPENAI_API_KEY')
rag = create_interactive_session('data/embeddings', api_key)

# Ask a question
result = rag.answer_question("Чингис хаан хэзээ төрсөн бэ?")
print(result['answer'])
```

### Advanced Configuration
```python
from mongolian_rag import MongolianHistoryRAG, RAGConfig

# Custom configuration
config = RAGConfig(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model='gpt-4o-mini',
    temperature=0.2,
    retrieval_top_k=7,
    language='mongolian'
)

# Create RAG agent
rag = MongolianHistoryRAG(config, 'data/embeddings')

# Ask questions
result = rag.answer_question("XIII зууны Монголын түүхийн талаар ярина уу?")
```

### Interactive Mode
```bash
python -m mongolian_rag.rag_agent
```

### Command Line Search
```bash
# Search for content
python -m mongolian_rag.retrieval_engine --query "Чингис хаан" --top-k 5

# Filter by period
python -m mongolian_rag.retrieval_engine --query "хувьсгал" --period "XX зуун"

# Filter by source
python -m mongolian_rag.retrieval_engine --query "нууц товчоо" --source "Монголын нууц товчоо"
```

## Configuration Options

### EmbeddingConfig
```python
@dataclass
class EmbeddingConfig:
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    chunk_size: int = 1000          # Characters per chunk
    chunk_overlap: int = 200        # Overlap between chunks
    batch_size: int = 100           # Embeddings per batch
    index_type: str = "IVF"         # IVF, Flat, or HNSW
    output_dir: str = "data/embeddings"
```

### RAGConfig
```python
@dataclass
class RAGConfig:
    openai_api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 1500
    retrieval_top_k: int = 5
    max_context_tokens: int = 3000
    language: str = "mongolian"     # mongolian, english, or auto
```

### SearchConfig
```python
@dataclass
class SearchConfig:
    top_k: int = 5
    score_threshold: float = 0.0
    rerank: bool = True
    deduplicate: bool = True
    max_chunk_per_entry: int = 3
```

## Dataset Statistics

Based on the cleaned unified dataset:

### Content Distribution
- **Total entries**: 44 historical documents
- **Total chunks**: 749 searchable segments
- **Average chunk length**: 907 characters
- **Embedding dimension**: 1536 (text-embedding-3-small)

### Period Coverage
- **XIII зуун (13th century)**: 186 chunks (24.8%)
- **XVII-XIX зуун (17th-19th centuries)**: 437 chunks (58.3%)
- **XX зуун (20th century)**: 118 chunks (15.8%)
- **Орчин үе (Modern era)**: 6 chunks (0.8%)
- **Эртний үе (Ancient era)**: 2 chunks (0.3%)

### Source Distribution
- **Монголын түүх, соёл, ёс заншил**: 553 chunks (73.8%)
- **Монголын нууц товчоо**: 183 chunks (24.4%)
- **mn.wikipedia.org**: 12 chunks (1.6%)
- **num.edu.mn**: 1 chunk (0.1%)

## Performance Metrics

### Embedding Generation
- **Model**: text-embedding-3-small
- **Dimension**: 1536
- **Processing speed**: ~50 texts per batch
- **Token limit**: 8,191 tokens per text

### FAISS Index
- **Index type**: IVF (Inverted File)
- **Search speed**: Sub-millisecond for top-k retrieval
- **Memory usage**: ~4.6MB for 749 vectors
- **Accuracy**: Approximate search with 95%+ recall

### RAG Performance
- **Response time**: 2-5 seconds per question
- **Context window**: Up to 3,000 tokens
- **Source attribution**: Automatic citation generation
- **Confidence scoring**: Based on retrieval similarity

## API Reference

### MongolianHistoryEmbedder
```python
class MongolianHistoryEmbedder:
    def __init__(self, config: EmbeddingConfig)
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]
    def create_embeddings(self, dataset_path: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]
    def save_pipeline(self, index: faiss.Index, metadata: List[Dict[str, Any]])
```

### MongolianHistoryRetriever
```python
class MongolianHistoryRetriever:
    def __init__(self, embeddings_dir: str, openai_api_key: str)
    def search(self, query: str, config: SearchConfig = None) -> List[RetrievalResult]
    def search_by_period(self, period: str, query: str = "", top_k: int = 10) -> List[RetrievalResult]
    def search_by_source(self, source: str, query: str = "", top_k: int = 10) -> List[RetrievalResult]
    def get_context_for_rag(self, query: str, max_tokens: int = 4000) -> Tuple[str, List[Dict[str, Any]]]
```

### MongolianHistoryRAG
```python
class MongolianHistoryRAG:
    def __init__(self, config: RAGConfig, embeddings_dir: str)
    def answer_question(self, question: str, use_history: bool = True) -> Dict[str, Any]
    def ask_about_period(self, period: str, question: str = "") -> Dict[str, Any]
    def ask_about_person(self, person: str, question: str = "") -> Dict[str, Any]
    def ask_about_event(self, event: str, question: str = "") -> Dict[str, Any]
    def export_conversation(self, filepath: str)
```

## File Structure

```
mongolian_rag/
├── __init__.py                 # Package initialization
├── embedding_pipeline.py      # Embedding generation and FAISS indexing
├── retrieval_engine.py        # Similarity search and retrieval
└── rag_agent.py               # Question-answering system

data/
├── mongolian_history_unified.jsonl    # Cleaned dataset
└── embeddings/                        # Generated embeddings
    ├── mongolian_history.faiss        # FAISS index
    ├── metadata.json                  # Chunk metadata
    ├── config.json                    # Pipeline configuration
    └── pipeline_summary.json          # Creation summary

scripts/
├── setup_rag_pipeline.py      # Complete setup script
├── demo_rag_pipeline.py       # Demo without API key
└── clean_and_merge_json.py    # Data cleaning script

examples/
├── basic_usage.py             # Basic usage example
└── advanced_usage.py          # Advanced configuration example
```

## Troubleshooting

### Common Issues

#### 1. OpenAI API Key Not Found
```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### 2. Dataset Not Found
```bash
python scripts/clean_and_merge_json.py
```

#### 3. FAISS Installation Issues
```bash
# For CPU-only systems
pip install faiss-cpu

# For GPU systems (if available)
pip install faiss-gpu
```

#### 4. Memory Issues with Large Datasets
- Reduce `batch_size` in EmbeddingConfig
- Use `chunk_size` to control memory usage
- Consider using FAISS IVF index for large datasets

#### 5. Slow Retrieval Performance
- Use IVF or HNSW index types for large datasets
- Adjust `nprobe` parameter for speed/accuracy tradeoff
- Consider GPU acceleration for very large datasets

### Performance Optimization

#### For Large Datasets (>10,000 chunks)
```python
config = EmbeddingConfig(
    index_type="IVF",
    nlist=100,           # Increase for larger datasets
    batch_size=50,       # Reduce if memory issues
    chunk_size=800       # Smaller chunks for better granularity
)
```

#### For Fast Retrieval
```python
config = EmbeddingConfig(
    index_type="HNSW",   # Fastest search
    chunk_size=1200      # Larger chunks for fewer vectors
)
```

#### For High Accuracy
```python
config = EmbeddingConfig(
    index_type="Flat",   # Exact search
    chunk_overlap=300    # More overlap for better coverage
)
```

## Demo Mode

For testing without an OpenAI API key:

```bash
python scripts/demo_rag_pipeline.py
```

This creates mock embeddings that demonstrate the pipeline functionality without requiring API access.

## Future Enhancements

### Planned Features
1. **Multi-modal Support**: Add image and document processing
2. **Advanced Filtering**: Date range, geographic, and thematic filters
3. **Batch Processing**: Bulk question processing capabilities
4. **Export Formats**: PDF, Word, and structured data exports
5. **Web Interface**: Browser-based question-answering interface

### Integration Opportunities
1. **Chatbot Integration**: Discord, Telegram, or web chat
2. **API Service**: REST API for external applications
3. **Mobile App**: Native mobile question-answering app
4. **Educational Platform**: Integration with learning management systems

## Contributing

### Adding New Data Sources
1. Convert data to JSONL format matching the schema
2. Run the cleaning script to merge with existing data
3. Regenerate embeddings with the updated dataset

### Improving Retrieval Quality
1. Experiment with different chunking strategies
2. Adjust embedding model parameters
3. Implement custom reranking algorithms
4. Add domain-specific preprocessing

### Extending Language Support
1. Add translation capabilities
2. Implement multilingual embeddings
3. Create language-specific system prompts

## License and Usage

This RAG system is designed for educational and research purposes. Please ensure compliance with OpenAI's usage policies and respect the intellectual property rights of the historical sources included in the dataset.

## Support

For technical issues or questions:
1. Check the troubleshooting section above
2. Review the example scripts in the `examples/` directory
3. Run the demo script to verify system functionality
4. Consult the API reference for detailed usage information