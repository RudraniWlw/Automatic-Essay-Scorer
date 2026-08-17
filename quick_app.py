from flask import Flask, request
import pickle
import re

app = Flask(__name__)

with open('essay_scorer_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_essay(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    essay = ''
    result = ''
    
    if request.method == 'POST':
        essay = request.form.get('essay', '')
        
        if essay and len(essay) > 10:
            features = vectorizer.transform([clean_essay(essay)])
            score_12 = model.predict(features)[0]
            score_10 = (score_12 / 12) * 10
            score_10 = max(1, min(10, score_10))
            
            if score_10 >= 8: 
                quality = "🌟 HIGH QUALITY"
                color = "#2d6a4f"
            elif score_10 >= 5: 
                quality = "📘 MEDIUM QUALITY"
                color = "#b9770e"
            else: 
                quality = "📕 LOW QUALITY"
                color = "#a4332a"
            
            bar_width = (score_10 / 10) * 100
            
            result = f'''
            <div style="text-align:center;padding:20px;">
                <h2 style="font-size:60px;margin:10px 0;color:#2c1810;">{score_10:.1f}<span style="font-size:24px;color:#6b4c3b;"> / 10</span></h2>
                <div style="width:100%;height:8px;background:#e8dcc8;border-radius:4px;margin:15px 0;">
                    <div style="width:{bar_width}%;height:100%;background:#2c1810;border-radius:4px;"></div>
                </div>
                <h3 style="color:{color};">{quality}</h3>
            </div>
            '''
    
    # HTML with essay directly embedded
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Essay Scorer</title>
        <style>
            body{{font-family:Georgia;background:#e8dcc8;padding:40px;}}
            .box{{max-width:700px;margin:0 auto;background:#f5ede1;padding:40px;border-radius:4px;border:1px solid #d4c5b2;}}
            h1{{text-align:center;color:#2c1810;border-bottom:3px double #2c1810;padding-bottom:15px;}}
            textarea{{width:100%;height:200px;padding:15px;font-size:16px;border:1px solid #c4b5a2;border-radius:2px;background:#fcf8f0;}}
            button{{background:#2c1810;color:white;padding:15px 50px;border:none;cursor:pointer;font-size:18px;display:block;margin:15px auto;}}
            button:hover{{background:#1a0f0a;}}
            .result{{margin-top:25px;background:#fcf8f0;border:1px solid #d4c5b2;padding:5px 20px;}}
            .tips{{margin-top:25px;padding:15px;background:#f5ede1;border-left:4px solid #2c1810;}}
            .footer{{text-align:center;margin-top:20px;color:#8b6b5a;font-size:0.8em;}}
        </style>
    </head>
    <body>
    <div class="box">
        <h1>📰 Essay Scorer</h1>
        <form method="POST">
            <textarea name="essay" placeholder="Paste your essay here...">{essay}</textarea>
            <button type="submit">✦ Score Essay ✦</button>
        </form>
        <div class="result">{result}</div>
        <div class="tips">
            <strong>✧ Tips:</strong>
            <ul>
                <li>Be specific with details</li>
                <li>Use evidence and examples</li>
                <li>Stay focused on your thesis</li>
            </ul>
        </div>
        <div class="footer">✦ Well written is well thought. ✦</div>
    </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)