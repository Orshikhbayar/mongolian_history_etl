# Fix OpenAI API Key & Create Embeddings

## Problem
Your current OpenAI API key is invalid or expired, preventing embedding creation.

## Solution

### Option 1: Interactive Setup (Recommended)

Run the setup script that will guide you through the process:

```bash
python setup_api_and_embeddings.py
```

This script will:
1. ✅ Test your current API key
2. ✅ Prompt for a new key if needed
3. ✅ Validate the new key
4. ✅ Save it to `.env` file (optional)
5. ✅ Create embeddings automatically

### Option 2: Manual Setup

#### Step 1: Get a Valid API Key

1. Go to: https://platform.openai.com/api-keys
2. Sign in to your OpenAI account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

#### Step 2: Set the API Key

**Option A: Environment Variable (Temporary)**
```bash
export OPENAI_API_KEY='your-new-api-key-here'
```

**Option B: .env File (Permanent)**
Create or edit `.env` file:
```bash
echo "OPENAI_API_KEY=your-new-api-key-here" > .env
```

Then load it:
```bash
source .env
```

#### Step 3: Create Embeddings

```bash
PYTHONPATH=. python scripts/setup_rag_pipeline.py
```

## What Gets Created

After successful setup:

```
data/
├── embeddings_filtered/          # New embeddings directory
│   ├── mongolian_history.faiss   # FAISS vector index
│   ├── metadata.json              # Document metadata
│   └── config.json                # Pipeline configuration
```

## Verify It Works

Test the RAG system:

```bash
PYTHONPATH=. python scripts/demo_rag_pipeline.py
```

Or use the simple version (no embeddings needed):

```bash
python demo_simple_rag.py
```

## Common Issues

### Issue 1: "Invalid API key"
**Solution**: Make sure you copied the entire key including the `sk-` prefix

### Issue 2: "Module not found: mongolian_rag"
**Solution**: Run with PYTHONPATH:
```bash
PYTHONPATH=. python setup_api_and_embeddings.py
```

### Issue 3: "Rate limit exceeded"
**Solution**: Wait a few minutes and try again, or use a different API key

### Issue 4: "Insufficient credits"
**Solution**: Add credits to your OpenAI account at https://platform.openai.com/account/billing

## Cost Estimate

Creating embeddings for the filtered dataset (21 documents):
- Model: `text-embedding-3-small`
- Estimated tokens: ~15,000
- Cost: ~$0.002 (less than 1 cent)

## Alternative: Use Without Embeddings

If you don't want to use OpenAI embeddings, you can use the simple RAG demo:

```bash
python demo_simple_rag.py
```

This uses basic text search and doesn't require:
- ❌ OpenAI API key
- ❌ Embeddings
- ❌ Any costs

But provides:
- ✅ Basic search functionality
- ✅ Source attribution
- ✅ Interactive Q&A

## Next Steps

Once embeddings are created:

1. **Test the RAG system**:
   ```bash
   PYTHONPATH=. python scripts/demo_rag_pipeline.py
   ```

2. **Use in your application**:
   ```python
   from mongolian_rag import create_interactive_session
   
   rag = create_interactive_session('data/embeddings_filtered', api_key)
   answer = rag.answer_question("Чингис хаан хэзээ төрсөн бэ?")
   ```

3. **Integrate with GPT-4** for better answers:
   ```python
   from mongolian_rag import MongolianHistoryRAG, RAGConfig
   
   config = RAGConfig(
       openai_api_key=api_key,
       model='gpt-4o',  # or gpt-4o-mini
       temperature=0.2
   )
   
   rag = MongolianHistoryRAG(config)
   ```

## Support

If you continue having issues:
1. Check your OpenAI account has credits
2. Verify the API key is active
3. Try creating a new API key
4. Use the simple RAG demo as a fallback
