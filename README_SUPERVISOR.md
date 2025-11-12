# Mongolian History RAG System - For Supervisor

## 🎯 Quick Start (2 Minutes)

### Option 1: Without API Key (Recommended for Testing)
```bash
# Install dependencies
pip install -r requirements.txt

# Run Simple RAG (no API key needed)
python demo_simple_rag.py
```

### Option 2: With Your Own API Key (Full Features)
```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive setup
python setup.py

# Start web interface
python preview_ui.py
```
Then open: http://localhost:5000

## 🔑 API Key - Three Options

### Option A: Use Your Own Key (Recommended)
1. Get free key: https://platform.openai.com/api-keys
2. Run: `python setup.py`
3. Enter your key when prompted
4. Cost: ~$0.0001 per question (essentially free)

### Option B: No API Key (Limited Features)
- Use Simple RAG: `python demo_simple_rag.py`
- Text-based search only
- No GPT-generated answers
- Completely free

### Option C: Request Key from Student
- Student can provide temporary key
- Share via secure channel (email/Slack)
- Not recommended for security reasons

## 📁 What's Included

### Core Files
- `demo_simple_rag.py` - Works without API key
- `quick_rag_demo.py` - GPT-powered Q&A
- `rag_with_gpt.py` - Full RAG system
- `preview_ui.py` - Web interface

### Data
- `data/mongolian_history_unified_filtered.jsonl` - 21 historical documents
- Sources: Wikipedia, textbooks, university content
- Excludes "Secret History" for modern relevance

### Configuration
- `.env.example` - API key template
- `setup.py` - Interactive setup
- `requirements.txt` - Dependencies

## 🧪 Testing the System

### Test 1: Simple RAG (No API Key)
```bash
python demo_simple_rag.py
```
Try asking: "Чингис хаан хэзээ төрсөн бэ?"

### Test 2: Web Interface (With API Key)
```bash
python preview_ui.py
```
Open http://localhost:5000 and try questions

### Test 3: Dataset Inspection
```bash
python -c "
import json
with open('data/mongolian_history_unified_filtered.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        doc = json.loads(line)
        print(f'{i}. {doc[\"source\"]} - {doc[\"period\"]}')
        if i >= 5: break
"
```

## 💡 Example Questions

Try these in any RAG system:
1. Чингис хаан хэзээ төрсөн бэ?
2. Монголын ардчилсан хувьсгал хэзээ болсон бэ?
3. Өгэдэй хааны тухай хэлнэ үү?
4. Их Монгол Улс хэзээ байгуулагдсан бэ?

## 📊 System Comparison

| Feature | Simple RAG | GPT-Powered RAG |
|---------|-----------|-----------------|
| API Key | ❌ Not needed | ✅ Required |
| Cost | Free | ~$0.0001/question |
| Answer Quality | Good | Excellent |
| Search Type | Text matching | Semantic |
| Offline | ✅ Yes | ❌ No |

## 🎨 Web Interface Features

- 🇲🇳 Mongolian language UI
- 🎨 Beautiful purple gradient design
- 💬 Chat-like interface
- 📚 Source citations
- ⚡ Real-time responses
- 💡 Example questions

## 🔒 Security Notes

- `.env` file is gitignored (not in repository)
- API keys never committed to git
- Each user should use their own key
- Keys can be rotated anytime

## 📈 Project Structure

```
3.llm_grpo_mgl/
├── api/                    # Vercel deployment
├── data/                   # Historical dataset
├── demo_simple_rag.py      # No API key needed ⭐
├── quick_rag_demo.py       # GPT-powered
├── rag_with_gpt.py         # Full system
├── preview_ui.py           # Web interface
├── setup.py                # Interactive setup ⭐
├── requirements.txt        # Dependencies
├── .env.example            # API key template
└── README_SUPERVISOR.md    # This file
```

## 🚀 Deployment Options

### Local (Current)
- Run on your machine
- Use for testing/demo
- No deployment needed

### Vercel (Production)
- Deploy to cloud
- Public URL
- Free tier available
- See: `VERCEL_DEPLOY.md`

## 💰 Cost Breakdown

### Development/Testing
- Simple RAG: **$0** (free)
- Local testing: **$0** (free)

### With OpenAI API
- API key: **Free** to create
- Per question: **~$0.0001** (0.01 cents)
- 100 questions: **~$0.01** (1 cent)
- 1000 questions: **~$0.10** (10 cents)

**Essentially free for evaluation purposes!**

## 🐛 Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "API key not found"
```bash
python setup.py
```

### "Dataset not found"
Make sure you're in the correct directory:
```bash
ls data/mongolian_history_unified_filtered.jsonl
```

### "Port already in use"
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9
```

## 📞 Contact

If you have questions:
1. Check `SETUP_GUIDE.md` for detailed instructions
2. Try Simple RAG first (no API key needed)
3. Contact student for assistance

## ✅ Evaluation Checklist

- [ ] System runs without errors
- [ ] Simple RAG works (no API key)
- [ ] Dataset loads correctly
- [ ] Answers are relevant
- [ ] Sources are cited
- [ ] UI is functional (if using API key)
- [ ] Code is well-documented
- [ ] Security best practices followed

## 🎓 Summary

This is a **production-ready RAG system** for Mongolian historical Q&A with:
- ✅ Multiple options (with/without API key)
- ✅ Clean, documented code
- ✅ Filtered dataset (21 documents)
- ✅ Beautiful web interface
- ✅ Secure API key handling
- ✅ Easy setup and testing

**Recommended:** Start with Simple RAG (no API key) to test functionality, then try GPT-powered version if you want to see full features.
