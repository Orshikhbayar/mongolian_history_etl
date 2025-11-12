# Filtered Dataset Information

## Overview
Created a filtered version of the unified dataset that excludes "Монголын нууц товчоо" (Secret History of the Mongols).

## Files

### Original Dataset
- **File**: `data/mongolian_history_unified.jsonl`
- **Total**: 44 documents
- **Sources**: All historical sources including Secret History

### Filtered Dataset
- **File**: `data/mongolian_history_unified_filtered.jsonl`
- **Total**: 21 documents
- **Excluded**: 23 documents from "Монголын нууц товчоо"

## Remaining Sources

| Source | Documents | Description |
|--------|-----------|-------------|
| mn.wikipedia.org | 12 | Wikipedia articles on Mongolian history |
| Монголын түүх, соёл, ёс заншил | 8 | Mongolian History, Culture, and Traditions textbook |
| num.edu.mn | 1 | National University of Mongolia content |

## Usage

### Simple RAG Demo
```bash
python demo_simple_rag.py
```

This demo uses the filtered dataset by default and provides:
- Text-based search (no embeddings needed)
- Interactive Q&A
- Source attribution
- Works without OpenAI API key

### For Full RAG System
When you have a valid OpenAI API key, update the embedding pipeline:

```python
# In scripts/setup_rag_pipeline.py or mongolian_rag/embedding_pipeline.py
dataset_path = 'data/mongolian_history_unified_filtered.jsonl'
```

## Why Filter Out Secret History?

The Secret History (Монголын нууц товчоо) is:
- Written in classical/archaic Mongolian
- Uses different vocabulary and grammar
- May not be relevant for modern historical questions
- Can be added back anytime if needed

## Adding It Back

To include Secret History again, simply use the original file:
```python
dataset_path = 'data/mongolian_history_unified.jsonl'
```

## Statistics

### Content Distribution
- **Modern sources**: 13 documents (Wikipedia + num.edu.mn)
- **Textbook content**: 8 documents
- **Total**: 21 documents covering various periods

### Period Coverage
- Эртний үе (Ancient period)
- XIII зуун (13th century)
- XVII-XIX зуун (17th-19th centuries)
- XX зуун (20th century)

## Next Steps

1. ✅ Filtered dataset created
2. ✅ Simple RAG demo working
3. ⏳ Get valid OpenAI API key for full RAG system
4. ⏳ Create embeddings with filtered dataset
5. ⏳ Deploy full RAG system with GPT-4 integration
