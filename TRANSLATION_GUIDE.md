# Mongolian Dataset Translation Guide

## Overview

The `translate_mixed_entries.py` script automatically detects and translates English or mixed-language text fields into Mongolian using OpenAI GPT-4o-mini API. This ensures all historical dataset entries are in clean Mongolian text before training.

## Features

### 🔍 **Language Detection**
- **Mongolian Cyrillic Pattern**: `[А-ЯӨҮа-яөү]`
- **English Pattern**: `[A-Za-z]`
- **Configurable Threshold**: Default 20% English ratio triggers translation
- **Smart Analysis**: Considers character ratios and content structure

### 🌐 **Translation Engine**
- **OpenAI GPT-4o-mini**: Professional-grade translation
- **Academic Tone Preservation**: Maintains historical accuracy
- **Retry Logic**: Up to 3 attempts with exponential backoff
- **Error Handling**: Graceful failure recovery

### 📁 **Multi-Format Support**
- **JSON Arrays**: Standard JSON format
- **JSONL**: Line-by-line JSON objects
- **GRPO Format**: Preference pairs (`prompt`/`chosen`/`rejected`)
- **Flexible Fields**: `text`, `content`, `title`, etc.

## Usage

### Basic Usage
```bash
# Translate all default datasets
python scripts/translate_mixed_entries.py

# Translate specific files
python scripts/translate_mixed_entries.py --files data/mixed_dataset.json

# Custom output and threshold
python scripts/translate_mixed_entries.py \\
  --files data/input.jsonl \\
  --output data/translated.jsonl \\
  --threshold 0.15
```

### Command Line Options
```bash
--files FILE [FILE...]     # Specific files to translate
--output PATH              # Output file path (default: data/mgl_history_translated.jsonl)
--threshold FLOAT          # English ratio threshold (default: 0.2)
--log-file PATH            # Log file path (default: data/mgl_history_translated.log)
```

## Setup

### Prerequisites
```bash
# Install dependencies
pip install openai tqdm

# Set OpenAI API key
export OPENAI_API_KEY=your_openai_api_key_here
```

### API Key Configuration
```bash
# Option 1: Environment variable
export OPENAI_API_KEY='your_openai_api_key_here'

# Option 2: Add to ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="your_openai_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

## Translation Workflow

### 1️⃣ **Load and Detect**
```python
# Language detection algorithm
english_chars = len(re.findall(r'[A-Za-z]', text))
mongolian_chars = len(re.findall(r'[А-ЯӨҮа-яөү]', text))
english_ratio = english_chars / (english_chars + mongolian_chars)

# Translation trigger
needs_translation = english_ratio >= threshold  # Default: 0.2
```

### 2️⃣ **Translate with GPT-4o-mini**
```python
# Translation prompt
system_prompt = """You are a professional Mongolian translator.
Translate into clear, natural Mongolian, preserving academic tone 
and factual meaning. Use proper Mongolian Cyrillic script."""

user_prompt = f"Translate the following text into Mongolian:\\n{text}"
```

### 3️⃣ **Validate and Save**
```python
# Translation validation
mongolian_ratio = mongolian_chars / total_chars
is_valid = mongolian_ratio >= 0.8  # 80% Mongolian after translation
```

## Example Results

### Input Dataset
```json
[
  {
    "text": "1990 оны ардчилсан хувьсгал нь Монгол Улсын түүхэнд чухал үйл явдал болсон.",
    "period": "XX зуун"
  },
  {
    "text": "Mining industry became crucial for Mongolia's economy in the 2000s.",
    "period": "XXI зуун"
  }
]
```

### Translation Output
```json
[
  {
    "text": "1990 оны ардчилсан хувьсгал нь Монгол Улсын түүхэнд чухал үйл явдал болсон.",
    "period": "XX зуун"
  },
  {
    "text": "Уул уурхайн салбар 2000-аад оноос эхлэн Монголын эдийн засгийн гол тулгуур болсон.",
    "period": "XXI зуун"
  }
]
```

### Summary Report
```
📊 MONGOLIAN TRANSLATION SUMMARY
==================================================
Input Processing:
  Total records processed: 2
  Mixed-language entries detected: 1
  Records skipped (already Mongolian): 1

Translation Results:
  Translation attempts: 1
  Successful translations: 1
  Failed translations: 0
  Success rate: 100.0%

Performance Metrics:
  Total processing time: 2.1s
  Average latency per entry: 2.1s
  Total API calls: 1
  Total tokens used: 45
  Total words processed: 12

Output Quality:
  Final Mongolian purity: 100.0%
  Output file: data/mgl_history_translated.jsonl
  Total output records: 2

✅ SUCCESS
```

## Supported Dataset Formats

### 1. Standard JSONL Format
```json
{"text": "Historical content", "period": "XIII зуун", "source": "Wikipedia"}
{"content": "Another entry", "date": "1206", "title": "Event Title"}
```

### 2. JSON Array Format
```json
[
  {"text": "First entry", "metadata": "..."},
  {"content": "Second entry", "metadata": "..."}
]
```

### 3. GRPO Preference Format
```json
{"prompt": "Question", "chosen": "Good answer", "rejected": "Bad answer"}
```

## Error Handling

### Common Issues and Solutions

#### 1. API Authentication Error
```bash
Error: Incorrect API key provided
Solution: Check your OPENAI_API_KEY environment variable
```

#### 2. Rate Limiting
```bash
Error: Rate limit exceeded
Solution: Script automatically retries with exponential backoff
```

#### 3. Translation Validation Failed
```bash
Warning: Translation validation failed
Solution: Script retries up to 3 times, logs failures for manual review
```

### Failed Entries Log
All failed translations are logged to `translate_failed.log`:
```
Failed Entry #1
File: data/input.json
Record Index: 5
Errors: ["Field 'text': Rate limit exceeded"]
Original Record: {...}
```

## Performance Optimization

### Batch Processing
```bash
# Process multiple files efficiently
python scripts/translate_mixed_entries.py \\
  --files data/*.json data/*.jsonl \\
  --output data/all_translated.jsonl
```

### Threshold Tuning
```bash
# More aggressive translation (15% English threshold)
python scripts/translate_mixed_entries.py --threshold 0.15

# Conservative translation (30% English threshold)  
python scripts/translate_mixed_entries.py --threshold 0.30
```

### Cost Management
- **Token Usage**: ~50-100 tokens per translation
- **API Calls**: 1 call per mixed-language field
- **Estimated Cost**: $0.001-0.002 per translation (GPT-4o-mini pricing)

## Quality Assurance

### Pre-Translation Validation
```bash
# Check dataset quality before translation
python scripts/validate_mgl_dataset.py --files data/input.json
```

### Post-Translation Validation
```bash
# Verify translation quality
python scripts/validate_mgl_dataset.py --files data/translated.jsonl
```

### Manual Review Process
1. **Check failed entries log** for problematic translations
2. **Sample validation** of successful translations
3. **Language purity verification** (should be >95%)
4. **Content accuracy review** for historical facts

## Integration with Training Pipeline

### Complete Workflow
```bash
# 1. Validate original dataset
python scripts/validate_mgl_dataset.py --files data/raw_dataset.json

# 2. Translate mixed-language entries
python scripts/translate_mixed_entries.py \\
  --files data/raw_dataset.json \\
  --output data/translated_dataset.jsonl

# 3. Validate translated dataset
python scripts/validate_mgl_dataset.py --files data/translated_dataset.jsonl

# 4. Proceed with training if quality is acceptable
if [ $? -eq 0 ]; then
    python train_model.py --data data/translated_dataset.jsonl
fi
```

### Automated Pipeline
```bash
#!/bin/bash
# translate_and_validate.sh

set -e

INPUT_FILE="$1"
OUTPUT_FILE="$2"

echo "🔄 Starting translation pipeline..."

# Translate
python scripts/translate_mixed_entries.py \\
  --files "$INPUT_FILE" \\
  --output "$OUTPUT_FILE"

# Validate
python scripts/validate_mgl_dataset.py \\
  --files "$OUTPUT_FILE" \\
  --output-report "${OUTPUT_FILE%.jsonl}_validation.txt"

echo "✅ Pipeline completed successfully"
```

## Demo Mode

For testing without API costs:
```bash
# Run demo with mock translations
python scripts/demo_translate_mixed_entries.py
```

The demo creates sample mixed-language data and shows the complete translation workflow with predefined translations.

## Best Practices

### Data Preparation
1. **Backup original datasets** before translation
2. **Use version control** to track changes
3. **Test with small samples** before full translation
4. **Monitor API usage** and costs

### Quality Control
1. **Set appropriate thresholds** based on dataset characteristics
2. **Review failed translations** manually
3. **Validate historical accuracy** of translations
4. **Check for cultural sensitivity** in translations

### Performance Tips
1. **Process during off-peak hours** to avoid rate limits
2. **Use batch processing** for multiple files
3. **Monitor token usage** to control costs
4. **Cache successful translations** to avoid re-processing

## Troubleshooting

### Common Problems

#### Script Hangs or Slow Performance
```bash
# Check API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \\
  https://api.openai.com/v1/models

# Reduce batch size or add delays
```

#### High Translation Failure Rate
```bash
# Check API key permissions
# Verify input data format
# Review error logs for patterns
```

#### Inconsistent Translation Quality
```bash
# Adjust temperature parameter (lower = more consistent)
# Review and refine system prompts
# Use manual validation samples
```

### Getting Help

1. **Check logs**: Review `mgl_history_translated.log` for detailed information
2. **Validate inputs**: Ensure input files are properly formatted
3. **Test API key**: Verify OpenAI API access and quotas
4. **Review documentation**: Check OpenAI API documentation for updates

## Conclusion

The Mongolian translation script provides a robust, automated solution for cleaning mixed-language datasets. With proper setup and monitoring, it ensures high-quality, pure Mongolian datasets ready for machine learning training.

Key benefits:
- **Automated language detection** and translation
- **Comprehensive error handling** and logging
- **Flexible format support** for various dataset types
- **Quality validation** and reporting
- **Cost-effective processing** with GPT-4o-mini

For optimal results, combine with the dataset validation tools and follow the recommended quality assurance processes.