# Setup Guide for Supervisor

## 🚀 Quick Start (3 Steps)

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone https://github.com/deeplearningmn/AIEng.git
cd AIEng/9.projects/3.llm_grpo_mgl

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Key

**Option A: Using .env file (Recommended)**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Change: OPENAI_API_KEY=your-openai-api-key-here
# To: OPENAI_API_KEY=sk-proj-...your-actual-key...
```

**Option B: Environment Variable (Temporary)**
```bash
export OPENAI_API_KEY='sk-proj-...your-key...'
```

**Option C: Interactive Prompt**
The system will ask for your API key when you run it if not configured.

### Step 3: Run the System

**Option 1: Simple RAG (No API Key Needed)**
```bash
python demo_simple_rag.py
```
- Works without OpenAI API key
- Uses text-based search
- Good for testing

**Option 2: Web Interface (Requires API Key)**
```bash
python preview_ui.py
```
Then open: http://localhost:5000

**Option 3: Command Line (Requires API Key)**
```bash
python quick_rag_demo.py
```

## 🔑 API Key Options for Supervisor

### Option 1: Provide Your Own Key (Recommended)
**Pros:**
- ✅ Full control
- ✅ Your own usage limits
- ✅ Secure

**How:**
1. Get key from: https://platform.openai.com/api-keys
2. Create `.env` file with your key
3. Run the system

**Cost:** ~$0.0001 per question (essentially free)

### Option 2: Use Simple RAG (No Key Needed)
**Pros:**
- ✅ No API key required
- ✅ Completely free
- ✅ Works offline

**Cons:**
- ❌ Text search only (not semantic)
- ❌ No GPT-generated answers

**How:**
```bash
python demo_simple_rag.py
```

### Option 3: Shared Key (Not Recommended)
**Cons:**
- ❌ Security risk
- ❌ Usage tracking issues
- ❌ Cost sharing problems

**If you must:**
- Share key via secure channel (not git)
- Use environment variable
- Monitor usage

### Option 4: Demo Mode
**For presentation only:**
```bash
# Uses cached/demo responses
python ui_preview.html  # Open in browser
```
Shows UI but doesn't make real API calls.

## 📋 What Supervisor Needs

### Minimum Requirements
- Python 3.8+
- pip
- Git

### For Full Functionality
- OpenAI API key
- Internet connection
- ~500MB disk space

### For Simple RAG Only
- Python 3.8+
- pip
- No API key needed
- Can work offline

## 🧪 Testing Without API Key

Your supervisor can test the system without an API key:

```bash
# 1. Simple RAG (text search)
python demo_simple_rag.py

# 2. UI Preview (static)
open ui_preview.html

# 3. Dataset exploration
python -c "
import json
with open('data/mongolian_history_unified_filtered.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        doc = json.loads(line)
        print(f'{i}. {doc[\"source\"]} - {doc[\"period\"]}')
        if i >= 5: break
"
```

## 🔒 Security Best Practices

### DO ✅
- Use `.env` file for API keys
- Add `.env` to `.gitignore`
- Share keys via secure channels (email, Slack DM)
- Use environment variables
- Rotate keys regularly

### DON'T ❌
- Commit API keys to git
- Share keys in public channels
- Hardcode keys in code
- Share keys in screenshots
- Post keys in issues/PRs

## 📁 Files to Share

### Include in Git
- ✅ All code files
- ✅ Dataset files
- ✅ `.env.example` (template)
- ✅ `.gitignore`
- ✅ Documentation
- ✅ Requirements

### Exclude from Git (in .gitignore)
- ❌ `.env` (contains real API key)
- ❌ `__pycache__/`
- ❌ `.venv/`
- ❌ Model files
- ❌ Embeddings

## 🎯 Recommended Approach for Supervisor

### For Review/Testing
```bash
# 1. Clone repo
git clone https://github.com/deeplearningmn/AIEng.git
cd AIEng/9.projects/3.llm_grpo_mgl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test without API key first
python demo_simple_rag.py

# 4. If they want full features, they add their own key
cp .env.example .env
# Edit .env with their key
python preview_ui.py
```

### For Presentation
```bash
# Option 1: Live demo with your key
export OPENAI_API_KEY='your-key'
python preview_ui.py

# Option 2: Static UI preview
open ui_preview.html

# Option 3: Simple RAG demo
python demo_simple_rag.py
```

## 💡 Alternative: Pre-generate Responses

If you don't want to share API key, you can pre-generate responses:

```bash
# Create a demo with cached responses
python -c "
import json

demo_qa = [
    {
        'question': 'Чингис хаан хэзээ төрсөн бэ?',
        'answer': 'Чингис хаан ойролцоогоор 1162 оны 11-р сард төрсөн...',
        'sources': ['Wikipedia', 'Textbook']
    },
    # Add more Q&A pairs
]

with open('demo_responses.json', 'w', encoding='utf-8') as f:
    json.dump(demo_qa, f, ensure_ascii=False, indent=2)
"
```

## 📞 Support

If supervisor has issues:

1. **No API key**: Use `demo_simple_rag.py`
2. **API key not working**: Check `.env` file format
3. **Dependencies missing**: Run `pip install -r requirements.txt`
4. **Dataset not found**: Ensure in correct directory

## ✅ Checklist Before Sharing

- [ ] `.gitignore` includes `.env`
- [ ] `.env.example` provided
- [ ] No API keys in code
- [ ] README.md updated
- [ ] SETUP_GUIDE.md included
- [ ] Simple RAG option available
- [ ] Requirements.txt complete
- [ ] Test without API key works

## 🎓 Summary

**Best approach:**
1. Push code to git (without API keys)
2. Include `.env.example` template
3. Supervisor adds their own API key
4. Or supervisor uses Simple RAG (no key needed)

**For demo/presentation:**
- Use your own API key temporarily
- Or use Simple RAG
- Or use static UI preview
