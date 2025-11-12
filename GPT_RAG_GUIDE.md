# GPT-Powered RAG System Guide

## Overview

Generate human-like answers to Mongolian history questions using GPT with your dataset as context.

## 🎯 What This Does

1. **Searches** your filtered dataset for relevant context
2. **Retrieves** the most relevant historical documents
3. **Generates** natural, human-like answers using GPT
4. **Cites** sources for transparency

## 📁 Files Created

| File | Purpose | Complexity |
|------|---------|------------|
| `quick_rag_demo.py` | Simple demo, easy to use | ⭐ Simple |
| `rag_with_gpt.py` | Full-featured RAG system | ⭐⭐⭐ Advanced |

## 🚀 Quick Start

### Option 1: Quick Demo (Recommended)

```bash
# Set your API key
export OPENAI_API_KEY='your-key-here'

# Run the demo
python quick_rag_demo.py
```

### Option 2: Full RAG System

```bash
# Set your API key
export OPENAI_API_KEY='your-key-here'

# Run full system
python rag_with_gpt.py
```

## 💡 Example Usage

### Input
```
❓ Чингис хаан хэзээ төрсөн бэ?
```

### Output
```
💬 ANSWER:
Чингис хаан ойролцоогоор 1162 оны 11-р сард төрсөн гэж үздэг. 
Тэрээр 1206 онд Монгол аймгуудыг нэгтгэж Их Монгол Улс буюу 
Монголын Эзэнт Гүрнийг байгуулсан Монголын их хаан байв. 
Хиад Боржигин Есүхэйн Тэмүжин нэртэй байсан бөгөөд 1227 оны 
5-р сарын 31-нд нас барсан.

📚 SOURCES:
1. mn.wikipedia.org - XIII зуун
2. mn.wikipedia.org - XIII зуун
3. mn.wikipedia.org - Эртний үе
```

## 🔧 Configuration

### Model Selection

**gpt-4o-mini** (Default)
- ✅ Fast
- ✅ Cheap (~$0.0001 per question)
- ✅ Good quality
- **Recommended for testing**

**gpt-4o**
- ✅ Best quality
- ✅ Better Mongolian understanding
- ❌ More expensive (~$0.001 per question)
- **Recommended for production**

**gpt-4-turbo**
- ✅ Balanced
- ✅ Good quality
- ✅ Reasonable cost
- **Good middle ground**

### Change Model

In `quick_rag_demo.py`:
```python
generate_answer(question, api_key, model="gpt-4o")
```

In `rag_with_gpt.py`:
```python
rag = MongolianRAGWithGPT(api_key=api_key, model="gpt-4o")
```

## 💰 Cost Estimate

### Per Question
- **gpt-4o-mini**: ~$0.0001 (0.01 cents)
- **gpt-4o**: ~$0.001 (0.1 cents)
- **gpt-4-turbo**: ~$0.0005 (0.05 cents)

### For 100 Questions
- **gpt-4o-mini**: ~$0.01 (1 cent)
- **gpt-4o**: ~$0.10 (10 cents)
- **gpt-4-turbo**: ~$0.05 (5 cents)

Very affordable for testing and production use!

## 🎨 Features

### Quick Demo (`quick_rag_demo.py`)
- ✅ Simple and fast
- ✅ Automatic context retrieval
- ✅ Source citations
- ✅ Interactive mode
- ✅ ~100 lines of code

### Full System (`rag_with_gpt.py`)
- ✅ Advanced search scoring
- ✅ Multiple language support
- ✅ Configurable temperature
- ✅ Detailed source metadata
- ✅ Error handling
- ✅ Model selection

## 📊 How It Works

```
User Question
     ↓
Search Dataset (text matching)
     ↓
Retrieve Top 3 Documents
     ↓
Format Context
     ↓
Send to GPT with System Prompt
     ↓
Generate Human-like Answer
     ↓
Display Answer + Sources
```

## 🔑 API Key Setup

### Temporary (Current Session)
```bash
export OPENAI_API_KEY='sk-...'
```

### Permanent (All Sessions)
```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
source ~/.zshrc
```

### In Script
```python
api_key = "sk-..."  # Not recommended for security
```

## 🧪 Testing

### Test with Sample Questions

```bash
python quick_rag_demo.py
```

Sample questions to try:
- Чингис хаан хэзээ төрсөн бэ?
- Монголын ардчилсан хувьсгал хэзээ болсон бэ?
- Өгэдэй хааны тухай хэлнэ үү?
- Их Монгол Улс хэзээ байгуулагдсан бэ?
- Монголын түүхийн чухал үйл явдлууд юу вэ?

## 🎯 Use Cases

### 1. Historical Q&A System
Build a chatbot that answers Mongolian history questions

### 2. Educational Tool
Help students learn Mongolian history interactively

### 3. Research Assistant
Quick lookup of historical facts with sources

### 4. Content Generation
Generate educational content based on historical sources

## 🔄 Comparison with Other Approaches

| Approach | Quality | Speed | Cost | Setup |
|----------|---------|-------|------|-------|
| Simple RAG (text search) | Good | Fast | Free | None |
| RAG + GPT (this) | Excellent | Fast | ~$0.0001/q | API key |
| Fine-tuned model | Excellent | Fast | High upfront | GPU + time |
| Embeddings + GPT | Best | Fast | ~$0.002 setup | API key |

## 🚨 Important Notes

### Data Privacy
- Your dataset is sent to OpenAI for context
- Only relevant excerpts (not full dataset)
- Consider data sensitivity

### API Limits
- Rate limits apply (check your OpenAI plan)
- Monitor usage in OpenAI dashboard
- Set up billing alerts

### Quality
- Answers depend on dataset quality
- GPT may occasionally hallucinate
- Always verify critical information

## 🛠️ Troubleshooting

### "Invalid API key"
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set it again
export OPENAI_API_KEY='your-key'
```

### "No relevant documents found"
- Try rephrasing your question
- Check if dataset contains relevant info
- Try broader search terms

### "Rate limit exceeded"
- Wait a few minutes
- Upgrade your OpenAI plan
- Use gpt-4o-mini (higher limits)

### "Module not found: openai"
```bash
pip install openai
```

## 📈 Next Steps

### 1. Test the System
```bash
python quick_rag_demo.py
```

### 2. Try Different Models
Compare gpt-4o-mini vs gpt-4o quality

### 3. Add More Data
Expand your filtered dataset for better coverage

### 4. Create Embeddings
For even better search quality:
```bash
python create_local_embeddings.py
```

### 5. Build an Application
Integrate into a web app or chatbot

## 🎓 Advanced Usage

### Custom System Prompt

```python
system_prompt = """Та Монголын түүхийн багш юм.
Оюутнуудад ойлгомжтой тайлбарлана уу."""

# Use in your code
```

### Adjust Temperature

```python
# More creative (0.7-1.0)
generate_answer(question, api_key, temperature=0.9)

# More factual (0.0-0.3)
generate_answer(question, api_key, temperature=0.2)
```

### Multiple Languages

The system auto-detects Mongolian vs English and adjusts accordingly.

## ✅ Summary

You now have a **GPT-powered RAG system** that:
- ✅ Generates human-like answers
- ✅ Uses your historical dataset
- ✅ Cites sources
- ✅ Works in Mongolian and English
- ✅ Costs ~$0.0001 per question
- ✅ Ready to use immediately

**Start with**: `python quick_rag_demo.py`
