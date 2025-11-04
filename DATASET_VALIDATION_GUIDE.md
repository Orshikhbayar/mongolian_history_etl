# Mongolian Dataset Validation Guide

## Overview

The `validate_mgl_dataset.py` script provides comprehensive validation for Mongolian historical datasets before GRPO (Generalized Reward-based Policy Optimization) or fine-tuning. It ensures data integrity, language consistency, and structural compliance.

## Features

### 🔍 **Comprehensive Validation**
- **Structure validation**: Required fields, data types, format compliance
- **Language detection**: Mongolian Cyrillic character analysis
- **Duplicate detection**: MD5 hash-based content deduplication
- **Mixed-language detection**: Identifies entries with >30% Latin text
- **Content analysis**: Character counts, token estimation, length statistics

### 📊 **Detailed Reporting**
- **Per-dataset statistics**: Validity rates, language purity, content metrics
- **Global summary**: Cross-dataset analysis and recommendations
- **Error logging**: Detailed invalid record tracking
- **Quality scoring**: Visual indicators for dataset quality

### 🛡️ **Robust Error Handling**
- **UTF-8 encoding**: Safe handling of Mongolian text
- **Graceful failures**: Continues validation on decode errors
- **Comprehensive logging**: All issues tracked and reported

## Usage

### Basic Usage
```bash
python scripts/validate_mgl_dataset.py
```

### Advanced Options
```bash
# Custom input directory
python scripts/validate_mgl_dataset.py --input-dir datasets

# Custom log file
python scripts/validate_mgl_dataset.py --output-log validation_errors.log

# Verbose logging
python scripts/validate_mgl_dataset.py --verbose
```

### Command Line Arguments
- `--input-dir`: Directory containing dataset files (default: `data`)
- `--output-log`: Output log file for invalid records (default: `data/invalid_records.log`)
- `--verbose`: Enable verbose logging for detailed debugging

## Supported File Formats

### Input Files
- **JSONL files**: One JSON object per line (`.jsonl`)
- **JSON files**: Single JSON array or object (`.json`)

### Supported Field Names
The validator recognizes multiple field naming conventions:
- `text`: Primary text content
- `content`: Alternative text field
- `prompt`: For instruction-following datasets
- `chosen`: For GRPO/preference datasets
- `rejected`: For GRPO/preference datasets
- `input`: For input-output datasets

### GRPO Format Support
```json
{"prompt": "Question or instruction", "chosen": "Preferred response", "rejected": "Alternative response"}
```

## Validation Criteria

### 1. Structure Validation
- ✅ Valid JSON format
- ✅ Required fields present
- ✅ Non-empty content
- ✅ Proper data types

### 2. Language Detection
- **Mongolian Pattern**: `[А-ЯӨҮа-яөү]` (Cyrillic + Mongolian letters)
- **Latin Pattern**: `[A-Za-z]` (English alphabet)
- **Purity Calculation**: `(Mongolian chars / Total letters) × 100`
- **Mixed-Language Threshold**: >30% Latin content

### 3. Content Analysis
- **Character counting**: Total text length
- **Token estimation**: Rough approximation (1 token ≈ 4 chars)
- **Length statistics**: Min, max, average content length
- **Duplicate detection**: MD5 hash comparison

### 4. Quality Thresholds
- **High Quality**: ≥95% validity, ≥95% Mongolian
- **Good Quality**: ≥90% validity, ≥90% Mongolian
- **Fair Quality**: ≥80% validity
- **Poor Quality**: <80% validity

## Output Format

### Console Report
```
🔍 Mongolian Dataset Validator
==================================================
📂 Input directory: data
📝 Log file: data/invalid_records.log

📁 Found 3 dataset files:
   - mgl_history_labeled.jsonl
   - modern_history_dataset.json
   - mgl_history_grpo.jsonl

📘 Dataset: mgl_history_labeled.jsonl
   Total records: 44
   Valid Mongolian entries: 44 (100.0%)
   Average length: 890 chars
   Longest: 5,407 chars | Shortest: 189 chars
   Language purity: 99.6% Mongolian

📗 Dataset: modern_history_dataset.json
   Total records: 20
   Valid Mongolian entries: 18 (90.0%)
   ⚠️  Mixed-language: 2
   💡 Recommendation: Review and clean mixed-language entries

============================================================
📊 VALIDATION SUMMARY
============================================================
📁 Files validated: 3
📄 Total records: 64
✅ Valid records: 62 (96.9%)
🇲🇳 Mongolian records: 62 (100.0%)
⚠️  Mixed-language records: 2

🏆 High-quality datasets: 2/3

💡 RECOMMENDATIONS:
   🧹 Clean mixed-language content in: modern_history_dataset.json
```

### Quality Indicators
- 📘 **Blue Book**: Excellent quality (≥95% validity & Mongolian)
- 📗 **Green Book**: Good quality (≥90% validity & Mongolian)
- 📙 **Orange Book**: Fair quality (≥80% validity)
- 📕 **Red Book**: Poor quality (<80% validity)

### Log File Format
```
2025-11-04 14:00:09 - INFO - dataset.jsonl[1]: Mixed language detected (purity: 20.5%)
2025-11-04 14:00:09 - WARNING - dataset.jsonl[3]: Empty or missing text content
2025-11-04 14:00:09 - WARNING - dataset.jsonl[4]: Missing required fields
```

## Validation Results

### Current Dataset Status
Based on validation of existing datasets:

#### ✅ **High-Quality Datasets**
- `mgl_history_labeled.jsonl`: 44 records, 97.4% purity
- `secret_history.jsonl`: 23 records, 95.4% purity
- `web_raw.jsonl`: 13 records, 99.8% purity
- `mongolian_history_textbook.jsonl`: 8 records, 99.3% purity

#### 📊 **Overall Statistics**
- **Total files**: 7 datasets
- **Total records**: 180 entries
- **Validity rate**: 100% (all records valid)
- **Mongolian coverage**: 100% (all records contain Mongolian)
- **Language purity**: 95.4-99.8% across datasets

#### ⚠️ **Issues Identified**
- **131 duplicate records** across datasets (expected due to merged files)
- **No mixed-language entries** in production datasets
- **No structural issues** in any dataset

## Integration with ML Pipelines

### Pre-Training Validation
```bash
# Validate before fine-tuning
python scripts/validate_mgl_dataset.py --input-dir training_data

# Check exit code for automation
if [ $? -eq 0 ]; then
    echo "✅ Data ready for training"
    python train_model.py
else
    echo "❌ Data validation failed"
    exit 1
fi
```

### GRPO Dataset Preparation
```bash
# Validate GRPO format
python scripts/validate_mgl_dataset.py --input-dir grpo_datasets --verbose

# Check for preference pairs
grep -c "chosen.*rejected" grpo_datasets/*.jsonl
```

### Continuous Integration
```yaml
# GitHub Actions example
- name: Validate Datasets
  run: |
    python scripts/validate_mgl_dataset.py --input-dir data
    if [ $? -ne 0 ]; then
      echo "Dataset validation failed"
      exit 1
    fi
```

## Common Issues and Solutions

### 1. Mixed-Language Content
**Issue**: Text contains significant Latin characters
```json
{"text": "This is English mixed with монгол хэл"}
```

**Solutions**:
- Translate English portions to Mongolian
- Remove or replace Latin text
- Split into separate language-specific datasets

### 2. Missing Required Fields
**Issue**: Records lack text content fields
```json
{"metadata": "info", "source": "web"}  // Missing 'text' field
```

**Solutions**:
- Add required `text`, `content`, or `prompt` fields
- Restructure data to include content fields
- Remove records without textual content

### 3. Empty Content
**Issue**: Text fields are empty or whitespace-only
```json
{"text": "", "source": "web"}
{"text": "   \n\n   ", "source": "web"}
```

**Solutions**:
- Remove empty records
- Populate with actual content
- Merge with related non-empty records

### 4. Encoding Issues
**Issue**: Mongolian characters not properly encoded
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solutions**:
- Ensure UTF-8 encoding: `--encoding utf-8`
- Convert files: `iconv -f ISO-8859-1 -t UTF-8`
- Use proper text editors that support UTF-8

### 5. Duplicate Content
**Issue**: Same content appears multiple times
```json
{"text": "Чингис хаан...", "source": "book1"}
{"text": "Чингис хаан...", "source": "book2"}  // Duplicate
```

**Solutions**:
- Use the cleaning script: `python scripts/clean_and_merge_json.py`
- Manual deduplication based on content hashes
- Keep highest quality version of duplicates

## Best Practices

### 1. Pre-Validation Preparation
- **Backup original data** before cleaning
- **Use UTF-8 encoding** for all text files
- **Standardize field names** across datasets
- **Remove obvious non-Mongolian content**

### 2. Validation Workflow
```bash
# 1. Initial validation
python scripts/validate_mgl_dataset.py --verbose

# 2. Clean identified issues
python scripts/clean_and_merge_json.py

# 3. Re-validate after cleaning
python scripts/validate_mgl_dataset.py

# 4. Proceed with training if validation passes
```

### 3. Quality Maintenance
- **Regular validation** during data collection
- **Automated checks** in CI/CD pipelines
- **Version control** for dataset changes
- **Documentation** of cleaning decisions

### 4. Performance Optimization
- **Batch processing** for large datasets
- **Parallel validation** for multiple files
- **Memory management** for very large files
- **Progress tracking** with tqdm

## Technical Implementation

### Language Detection Algorithm
```python
# Mongolian character pattern
MONGOLIAN_PATTERN = re.compile(r'[А-ЯӨҮа-яөү]')

# Purity calculation
mongolian_chars = len(MONGOLIAN_PATTERN.findall(text))
latin_chars = len(re.findall(r'[A-Za-z]', text))
purity = (mongolian_chars / (mongolian_chars + latin_chars)) * 100
```

### Duplicate Detection
```python
# Content normalization and hashing
normalized = re.sub(r'\s+', ' ', text.lower().strip())
content_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
```

### Token Estimation
```python
# Rough approximation for Mongolian text
estimated_tokens = len(text) // 4
```

## Dependencies

### Required Packages
- `json`: JSON parsing and validation
- `re`: Regular expression pattern matching
- `hashlib`: MD5 hash generation for duplicates
- `collections`: Counter and defaultdict utilities
- `pathlib`: Modern path handling
- `logging`: Comprehensive error logging
- `argparse`: Command-line interface
- `datetime`: Timestamp generation

### Optional Packages
- `tqdm`: Progress bars (graceful fallback if not available)

### Installation
```bash
# Core dependencies (usually pre-installed)
python -m pip install --upgrade pip

# Optional progress bars
pip install tqdm
```

## Exit Codes

The validator returns appropriate exit codes for automation:

- **0**: All datasets passed validation (ready for training)
- **1**: Issues found (mixed-language, low purity, structural problems)

## Future Enhancements

### Planned Features
1. **Advanced Language Detection**: Machine learning-based language identification
2. **Content Quality Scoring**: Semantic analysis of historical accuracy
3. **Automated Cleaning**: Suggestions for fixing common issues
4. **Batch Processing**: Parallel validation for large dataset collections
5. **Integration APIs**: REST endpoints for validation services

### Extensibility
The validator is designed to be easily extended:
- **Custom validation rules**: Add domain-specific checks
- **New file formats**: Support additional data formats
- **Language support**: Extend to other languages
- **Quality metrics**: Add custom quality assessments

## Support and Troubleshooting

### Common Error Messages

#### "No dataset files found"
- Check input directory path
- Ensure files have `.json` or `.jsonl` extensions
- Verify file permissions

#### "JSON decode error"
- Validate JSON syntax with online tools
- Check for UTF-8 encoding issues
- Look for unescaped quotes or special characters

#### "Missing required fields"
- Add `text`, `content`, or `prompt` fields
- Check field naming conventions
- Ensure fields contain actual content

### Getting Help
1. **Check the log file** for detailed error information
2. **Use verbose mode** (`--verbose`) for debugging
3. **Validate individual files** to isolate issues
4. **Review the validation criteria** in this guide

The Mongolian Dataset Validator ensures your historical datasets meet the highest standards for machine learning applications, providing confidence in data quality before expensive training processes.