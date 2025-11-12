"""
Vercel Serverless Function for Mongolian History RAG
"""

from flask import Flask, request, jsonify, render_template_string
import json
import os
from pathlib import Path

app = Flask(__name__)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="mn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Монголын Түүхийн Асуулт Хариулт</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        .input-group label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: #333;
        }
        
        .input-group textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            min-height: 100px;
            transition: border-color 0.3s;
        }
        
        .input-group textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 16px;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
            font-weight: 600;
            transition: transform 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        
        .result.show {
            display: block;
        }
        
        .result h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .answer {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            line-height: 1.6;
        }
        
        .sources {
            background: white;
            padding: 20px;
            border-radius: 10px;
        }
        
        .source-item {
            padding: 10px;
            border-left: 3px solid #667eea;
            margin-bottom: 10px;
            background: #f8f9fa;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        
        .error.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🇲🇳 Монголын Түүх</h1>
            <p>Монголын түүхийн тухай асуугаарай</p>
        </div>
        
        <div class="content">
            <div class="input-group">
                <label for="question">Асуулт:</label>
                <textarea 
                    id="question" 
                    placeholder="Жишээ: Чингис хаан хэзээ төрсөн бэ?"
                ></textarea>
            </div>
            
            <button class="btn" onclick="askQuestion()">Асуух</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px;">Хариулт бэлтгэж байна...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="result" id="result">
                <h3>💬 Хариулт:</h3>
                <div class="answer" id="answer"></div>
                
                <h3>📚 Эх сурвалж:</h3>
                <div class="sources" id="sources"></div>
            </div>
        </div>
    </div>
    
    <script>
        async function askQuestion() {
            const question = document.getElementById('question').value.trim();
            
            if (!question) {
                showError('Асуулт оруулна уу');
                return;
            }
            
            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');
            document.getElementById('error').classList.remove('show');
            document.querySelector('.btn').disabled = true;
            
            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    showResult(data);
                }
            } catch (error) {
                showError('Алдаа гарлаа: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('show');
                document.querySelector('.btn').disabled = false;
            }
        }
        
        function showResult(data) {
            document.getElementById('answer').textContent = data.answer;
            
            const sourcesDiv = document.getElementById('sources');
            sourcesDiv.innerHTML = '';
            
            data.sources.forEach((source, index) => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                sourceItem.innerHTML = `
                    <strong>${index + 1}. ${source.source}</strong><br>
                    ${source.period ? `Үе: ${source.period}<br>` : ''}
                    ${source.chapter ? `Бүлэг: ${source.chapter}` : ''}
                `;
                sourcesDiv.appendChild(sourceItem);
            });
            
            document.getElementById('result').classList.add('show');
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.classList.add('show');
        }
        
        // Allow Enter to submit (with Shift+Enter for new line)
        document.getElementById('question').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""

def search_documents(query, top_k=3):
    """Simple search for relevant documents."""
    dataset_path = Path(__file__).parent.parent / "data" / "mongolian_history_unified_filtered.jsonl"
    
    if not dataset_path.exists():
        return []
    
    documents = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    
    # Simple scoring
    query_lower = query.lower()
    scored = []
    
    for doc in documents:
        text = doc.get('text', '').lower()
        score = 0
        
        for word in query_lower.split():
            if word in text:
                score += text.count(word)
        
        if score > 0:
            scored.append((score, doc))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_k]]

def generate_answer_with_gpt(question, docs):
    """Generate answer using OpenAI GPT."""
    from openai import OpenAI
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None, "OpenAI API key not configured"
    
    client = OpenAI(api_key=api_key)
    
    # Prepare context
    context = "\n\n".join([
        f"[Эх сурвалж {i+1}]: {doc.get('text', '')[:600]}"
        for i, doc in enumerate(docs)
    ])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Та Монголын түүхийн мэргэжилтэн юм. Өгөгдсөн эх сурвалжид үндэслэн асуултад монгол хэлээр, байгалийн хүнлэг яриагаар хариулна уу."
                },
                {
                    "role": "user",
                    "content": f"Эх сурвалж:\n\n{context}\n\nАсуулт: {question}\n\nХариулт:"
                }
            ],
            temperature=0.7,
            max_tokens=400
        )
        
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, str(e)

@app.route('/')
def home():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/ask', methods=['POST'])
def ask():
    """API endpoint for questions."""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Асуулт оруулна уу'}), 400
    
    # Search for relevant documents
    docs = search_documents(question, top_k=3)
    
    if not docs:
        return jsonify({
            'answer': 'Уучлаарай, энэ асуултад хариулах мэдээлэл олдсонгүй.',
            'sources': []
        })
    
    # Generate answer with GPT
    answer, error = generate_answer_with_gpt(question, docs)
    
    if error:
        return jsonify({'error': f'Алдаа: {error}'}), 500
    
    # Prepare sources
    sources = []
    for doc in docs:
        sources.append({
            'source': doc.get('source', 'Unknown'),
            'period': doc.get('period', ''),
            'chapter': doc.get('chapter', '')
        })
    
    return jsonify({
        'answer': answer,
        'sources': sources
    })

# For Vercel
app = app
