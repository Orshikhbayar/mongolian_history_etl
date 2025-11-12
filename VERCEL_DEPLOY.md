# Deploy to Vercel

## Quick Deploy

### Option 1: Deploy via Vercel CLI (Recommended)

1. **Install Vercel CLI**
```bash
npm install -g vercel
```

2. **Login to Vercel**
```bash
vercel login
```

3. **Deploy**
```bash
vercel
```

4. **Set Environment Variable**
```bash
vercel env add OPENAI_API_KEY
# Paste your OpenAI API key when prompted
```

5. **Redeploy with Environment Variable**
```bash
vercel --prod
```

### Option 2: Deploy via Vercel Dashboard

1. **Go to** https://vercel.com/new

2. **Import your GitHub repository**
   - Connect your GitHub account
   - Select `mongolian_history_etl` repository

3. **Configure Project**
   - Framework Preset: Other
   - Root Directory: ./
   - Build Command: (leave empty)
   - Output Directory: (leave empty)

4. **Add Environment Variable**
   - Go to Settings → Environment Variables
   - Add: `OPENAI_API_KEY` = `your-api-key`

5. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete

## What Gets Deployed

- **Web Interface**: Beautiful Mongolian UI
- **API Endpoint**: `/api/ask` for questions
- **Dataset**: Filtered Mongolian history data
- **GPT Integration**: Human-like answers

## After Deployment

Your app will be available at:
```
https://your-project-name.vercel.app
```

## Testing

Visit your deployment URL and try:
- Чингис хаан хэзээ төрсөн бэ?
- Монголын ардчилсан хувьсгал хэзээ болсон бэ?
- Өгэдэй хааны тухай хэлнэ үү?

## Cost

- **Vercel**: Free tier (100GB bandwidth/month)
- **OpenAI**: ~$0.0001 per question
- **Total**: Essentially free for moderate use

## Troubleshooting

### "Module not found"
- Check `requirements.txt` is in root directory
- Redeploy: `vercel --prod`

### "API key not found"
- Add environment variable in Vercel dashboard
- Redeploy after adding

### "Dataset not found"
- Ensure `data/mongolian_history_unified_filtered.jsonl` is committed
- Check file path in `api/index.py`

## Custom Domain

1. Go to Vercel Dashboard → Your Project → Settings → Domains
2. Add your custom domain
3. Follow DNS configuration instructions

## Updates

To update your deployment:
```bash
git add .
git commit -m "Update"
git push origin main
```

Vercel will automatically redeploy!
