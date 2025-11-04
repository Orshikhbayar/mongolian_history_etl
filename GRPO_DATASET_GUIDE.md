# GRPO Dataset Builder Guide

## Overview

The `build_grpo_dataset.py` script generates GRPO (Generative Reinforcement Preference Optimization) datasets from Mongolian historical data. Each record contains a user question (prompt) and two model responses: a "chosen" (better) and a "rejected" (worse) answer for preference-based fine-tuning.

## Features

### 🎯 **GRPO Dataset Generation**
- **Automatic Question Generation**: Creates relevant historical questions from corpus content
- **Preference Pair Creation**: Generates "chosen" vs "rejected" response pairs
- **Quality Validation**: Ensures Mongolian language purity and content quality
- **Comprehensive Statistics**: Detailed metrics and quality reports

### 🧠 **Intelligent Question Generation**
- **Topic Extraction**: Identifies historical topics from content using regex patterns
- **Template-Based Questions**: Uses 15+ Mongolian question templates
- **Context Preservation**: Maintains factual context for accurate responses
- **Diversity Control**: Ensures varied question types and topics

### 🌐 **OpenAI Integration**
- **GPT-4o-mini**: Professional-grade response generation
- **Preference Optimization**: Creates clearly differentiated response pairs
- **Academic Tone**: Maintains historical accuracy and scholarly language
- **Error Recovery**: Robust retry logic with exponential backoff

## Usage

### Basic Usage
```bash
# Generate GRPO dataset from default sources
python scripts/build_grpo_dataset.py

# Use specific source files
python scripts/build_grpo_dataset.py --source data/mgl_history_translated.jsonl

# Custom configuration
python scripts/build_grpo_dataset.py \\
  --source data/corpus.jsonl \\
  --output data/custom_grpo.jsonl \\
  --pairs-per-topic 20
```

### Command Line Options
```bash
--source FILE [FILE...]        # Source data files (auto-detect if not specified)
--rag-log FILE                 # Existing RAG Q&A log file
--output FILE                  # Output GRPO dataset path (default: data/mgl_history_grpo.jsonl)
--stats FILE                   # Statistics output path (default: data/mgl_history_grpo_stats.json)
--pairs-per-topic INT          # Number of pairs to generate per topic (default: 15)
--log-file FILE                # Invalid entries log file (default: data/grpo_invalid.log)
```

## Dataset Generation Logic

### 1️⃣ **Load Source Data**
```python
# Supported formats
- JSONL: Line-by-line JSON objects
- JSON: Standard JSON arrays
- Fields: 'text', 'content', 'chosen' for content extraction

# Auto-detection priority
1. data/mgl_history_translated.jsonl  # Preferred (clean Mongolian)
2. data/mongolian_history_unified.jsonl
3. data/mgl_history_labeled.jsonl
```

### 2️⃣ **Question Generation**
```python
# Topic extraction patterns
topic_patterns = [
    r'(\\d{4})\\s*оны?\\s+([^.!?]+)',      # Year-based events
    r'([А-ЯӨҮ][а-яөү\\s]+(?:хувьсгал|дайн|хаан|улс))',  # Historical terms
    r'([А-ЯӨҮ][а-яөү\\s]+(?:үе|цаг|зуун))',             # Time periods
]

# Question templates
templates = [
    "{topic} хэзээ болсон бэ?",
    "{topic}-ын үндсэн шалтгаан юу байсан бэ?",
    "{topic}-ын үр дүн нь юу байсан бэ?",
    "{topic} яагаад чухал байсан бэ?",
    # ... 15+ templates total
]
```

### 3️⃣ **GRPO Pair Generation**
```python
# System prompt for GPT-4o-mini
system_prompt = """You are an expert in Mongolian history and language.
Create two different quality responses:
1. CHOSEN: Strong, factual, coherent (80-150 words)
2. REJECTED: Weak, vague, incomplete (40-80 words)

Requirements:
- Both responses in Mongolian
- CHOSEN should be comprehensive and accurate
- REJECTED should be obviously inferior but not completely wrong
- Maintain historical accuracy in CHOSEN response
"""

# API configuration
model = "gpt-4o-mini"
temperature = 0.4  # Balance between consistency and variation
max_tokens = 900
max_retries = 2
```

### 4️⃣ **Validation and Quality Control**
```python
# Validation criteria
- All required fields present: prompt, chosen, rejected
- Minimum length: 100 characters per field
- Mongolian purity: ≥80% Cyrillic characters
- Response differentiation: chosen ≠ rejected
- Content quality: Factual accuracy and coherence
```

## Example Output

### Generated GRPO Pair
```json
{
  "prompt": "Чингис хааны ач холбогдол юунд оршдог вэ?",
  "chosen": "Чингис хаан (1162-1227) бол Монголын агуу хаан, Их Монгол Улсыг байгуулагч юм. Тэрээр 1206 онд Монголын овог аймгуудыг нэгтгэж, дэлхийн түүхэн дэх хамгийн том эзэнт гүрнийг байгуулсан. Чингис хааны удирдлага дор Монголчууд Хятад, Хорезм, Орос зэрэг олон улсыг байлдан дагуулж, Монголын соёл, хууль тогтоомжийг дэлгэрүүлсэн.",
  "rejected": "Чингис хаан бол Монголын хаан байсан. Тэр маш хүчирхэг байсан бөгөөд олон газар байлдсан."
}
```

### Summary Report
```
📊 GRPO DATASET GENERATION REPORT
==================================================
Generation Results:
  Total prompts generated: 120
  Valid pairs: 112
  Rejected pairs: 8
  Success rate: 93.3%

Quality Metrics:
  Average prompt length: 8.2 words
  Average chosen length: 87.5 words
  Average rejected length: 23.1 words
  Dataset purity: 99.2% Mongolian

Performance:
  Total processing time: 145.7s
  Total API calls: 118
  Total tokens used: 15,247
  API failures: 3
  Validation failures: 5

Status: ✅ SUCCESS
✅ Ready for GRPO fine-tuning
```

## Quality Assurance

### Response Quality Differentiation
- **Chosen Responses**: 80-150 words, comprehensive, factually accurate
- **Rejected Responses**: 40-80 words, vague, incomplete information
- **Clear Preference**: Obvious quality difference for effective training

### Language Purity Standards
- **Mongolian Threshold**: ≥80% Cyrillic characters
- **Content Validation**: Historical accuracy and cultural appropriateness
- **Grammar Check**: Proper Mongolian language structure

### Error Handling
- **API Failures**: Automatic retry with exponential backoff
- **JSON Parsing**: Robust error recovery and logging
- **Validation Failures**: Detailed error reporting and filtering

## Integration with Training Pipeline

### GRPO Fine-tuning Workflow
```bash
# 1. Generate GRPO dataset
python scripts/build_grpo_dataset.py \\
  --source data/mgl_history_translated.jsonl \\
  --pairs-per-topic 25

# 2. Validate dataset quality
python scripts/validate_mgl_dataset.py \\
  --files data/mgl_history_grpo.jsonl

# 3. Train with GRPO
python train_grpo_model.py \\
  --dataset data/mgl_history_grpo.jsonl \\
  --model mongolian-llama-base
```

### Dataset Statistics Tracking
```json
{
  "total_prompts_generated": 120,
  "valid_pairs": 112,
  "rejected_pairs": 8,
  "avg_chosen_length": 87.5,
  "avg_rejected_length": 23.1,
  "mongolian_purity": 99.2,
  "processing_time": 145.7,
  "total_tokens_used": 15247
}
```

## Demo Mode

For testing without API costs:
```bash
# Run demo with mock responses
python scripts/demo_build_grpo_dataset.py
```

The demo creates realistic GRPO pairs using predefined high-quality responses for common Mongolian historical topics.

## Performance Optimization

### Cost Management
- **Token Efficiency**: ~120-150 tokens per GRPO pair
- **Batch Processing**: Efficient API usage patterns
- **Estimated Cost**: $0.002-0.003 per pair (GPT-4o-mini pricing)

### Quality vs Quantity Balance
- **Minimum Viable Dataset**: 50+ pairs for basic training
- **Recommended Size**: 200+ pairs for robust training
- **Optimal Range**: 500-1000 pairs for production models

### Processing Speed
- **Generation Rate**: ~2-3 pairs per minute (API dependent)
- **Batch Size**: Process 10-20 topics simultaneously
- **Retry Logic**: Exponential backoff prevents rate limiting

## Best Practices

### Dataset Preparation
1. **Use Clean Sources**: Prefer translated/validated Mongolian content
2. **Diverse Topics**: Ensure broad historical coverage
3. **Quality Control**: Review sample pairs before full generation
4. **Version Control**: Track dataset versions and generation parameters

### GRPO Training Considerations
1. **Preference Clarity**: Ensure obvious quality differences
2. **Response Length**: Maintain consistent length ratios
3. **Cultural Sensitivity**: Verify historical accuracy and cultural appropriateness
4. **Evaluation Metrics**: Plan for preference accuracy measurement

### Troubleshooting

#### Low Success Rate
```bash
# Check API key and quotas
# Review source data quality
# Adjust validation thresholds
# Increase retry attempts
```

#### Poor Response Quality
```bash
# Refine system prompts
# Adjust temperature settings
# Review topic extraction patterns
# Manual quality sampling
```

#### High API Costs
```bash
# Reduce pairs-per-topic
# Use more selective topic extraction
# Implement response caching
# Monitor token usage patterns
```

## Advanced Configuration

### Custom Question Templates
```python
# Add domain-specific templates
custom_templates = [
    "{topic}-ын эдийн засгийн нөлөө юу байсан бэ?",
    "{topic}-д ямар соёлын өөрчлөлт гарсан бэ?",
    "{topic}-ын олон улсын хариу үйлдэл хэрхэн байсан бэ?"
]
```

### Response Quality Tuning
```python
# Adjust response length requirements
chosen_min_words = 80
chosen_max_words = 150
rejected_min_words = 40
rejected_max_words = 80

# Modify quality differentiation
quality_gap_threshold = 2.0  # Chosen should be 2x better than rejected
```

## Conclusion

The GRPO dataset builder provides a complete solution for generating high-quality preference datasets for Mongolian language model fine-tuning. With intelligent question generation, robust quality control, and comprehensive error handling, it ensures production-ready datasets for reinforcement learning from human feedback (RLHF) training.

Key benefits:
- **Automated Generation**: Minimal manual intervention required
- **Quality Assurance**: Built-in validation and error handling
- **Scalable Processing**: Efficient API usage and batch processing
- **Cultural Accuracy**: Maintains Mongolian historical and linguistic authenticity
- **Training Ready**: Direct integration with GRPO fine-tuning pipelines