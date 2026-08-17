# ==================== DAY 3: WEB INTERFACE (NEWSPAPER THEME) ====================
from flask import Flask, request, render_template_string
import pickle
import re
import pandas as pd
from datetime import datetime

# Load the saved model
print("🚀 Loading Essay Scorer Model...")
with open('essay_scorer_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("✅ Model loaded successfully!")

# Initialize Flask app
app = Flask(__name__)

# Clean function
def clean_essay(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def score_essay(essay_text):
    """Predict score for any essay"""
    cleaned = clean_essay(essay_text)
    features = vectorizer.transform([cleaned])
    score = model.predict(features)[0]
    return score

# HTML Template - Newspaper Style
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>📰 The Essay Times - Automated Scoring</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Source+Serif+Pro:wght@400;600;700&family=Special+Elite&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: #e8dcc8;
            background-image: 
                linear-gradient(rgba(200, 180, 150, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(200, 180, 150, 0.1) 1px, transparent 1px);
            background-size: 20px 20px;
            min-height: 100vh;
            padding: 30px 20px;
            font-family: 'Source Serif Pro', Georgia, serif;
        }
        
        .newspaper-container {
            max-width: 900px;
            margin: 0 auto;
            background: #f5ede1;
            padding: 40px 45px;
            border-radius: 4px;
            box-shadow: 
                0 2px 20px rgba(0,0,0,0.15),
                inset 0 1px 0 rgba(255,255,255,0.8);
            border: 1px solid #d4c5b2;
            position: relative;
        }
        
        .newspaper-container::before {
            content: '';
            position: absolute;
            top: 20px;
            left: 20px;
            right: 20px;
            bottom: 20px;
            border: 1px solid rgba(180, 160, 140, 0.2);
            pointer-events: none;
            border-radius: 2px;
        }
        
        /* HEADER */
        .newspaper-header {
            text-align: center;
            border-bottom: 3px double #2c1810;
            padding-bottom: 20px;
            margin-bottom: 25px;
        }
        
        .publication-name {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 3.2em;
            font-weight: 900;
            color: #2c1810;
            letter-spacing: 8px;
            text-transform: uppercase;
            margin-bottom: 5px;
            text-shadow: 1px 1px 0 rgba(44, 24, 16, 0.1);
        }
        
        .publication-subtitle {
            font-family: 'Special Elite', cursive;
            font-size: 0.9em;
            color: #5c4033;
            letter-spacing: 4px;
            margin-bottom: 8px;
        }
        
        .edition-line {
            display: flex;
            justify-content: space-between;
            font-size: 0.75em;
            color: #6b4c3b;
            border-top: 1px solid #d4c5b2;
            padding-top: 8px;
            font-family: 'Special Elite', cursive;
            letter-spacing: 1px;
        }
        
        .edition-line span {
            background: #f5ede1;
            padding: 0 8px;
        }
        
        /* MAIN HEADLINE */
        .headline {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 2.4em;
            font-weight: 900;
            color: #1a0f0a;
            text-align: center;
            margin: 15px 0 10px 0;
            line-height: 1.2;
            letter-spacing: -0.5px;
        }
        
        .headline-sub {
            font-family: 'Special Elite', cursive;
            text-align: center;
            font-size: 1em;
            color: #6b4c3b;
            margin-bottom: 25px;
            font-style: italic;
        }
        
        .divider {
            border: none;
            border-top: 2px solid #2c1810;
            margin: 20px 0;
        }
        
        .divider-dotted {
            border: none;
            border-top: 1px dashed #b8a392;
            margin: 15px 0;
        }
        
        /* ESSAY INPUT AREA - NEWSPAPER COLUMN STYLE */
        .essay-section {
            margin: 20px 0 25px 0;
        }
        
        .section-title {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.3em;
            font-weight: 700;
            color: #2c1810;
            margin-bottom: 12px;
            letter-spacing: 1px;
        }
        
        .section-title::before {
            content: '✦ ';
            color: #8b6b5a;
        }
        
        textarea {
            width: 100%;
            min-height: 250px;
            padding: 20px 22px;
            background: #fcf8f0;
            border: 1px solid #c4b5a2;
            border-radius: 2px;
            font-family: 'Source Serif Pro', Georgia, serif;
            font-size: 16px;
            line-height: 1.8;
            color: #2c1810;
            resize: vertical;
            transition: all 0.3s ease;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }
        
        textarea:focus {
            outline: none;
            border-color: #2c1810;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.08), 0 0 0 3px rgba(44, 24, 16, 0.1);
        }
        
        textarea::placeholder {
            color: #b8a392;
            font-style: italic;
            font-size: 15px;
        }
        
        /* BUTTON - NEWSPAPER STYLE */
        .button-container {
            text-align: center;
            margin: 25px 0 20px 0;
        }
        
        .btn-score {
            background: #2c1810;
            color: #f5ede1;
            border: 1px solid #1a0f0a;
            padding: 16px 55px;
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.1em;
            font-weight: 700;
            letter-spacing: 3px;
            text-transform: uppercase;
            cursor: pointer;
            transition: all 0.3s ease;
            border-radius: 2px;
            position: relative;
        }
        
        .btn-score:hover {
            background: #1a0f0a;
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(44, 24, 16, 0.3);
        }
        
        .btn-score:active {
            transform: translateY(0);
        }
        
        /* RESULTS - NEWSPAPER ARTICLE STYLE */
        .result-section {
            margin-top: 30px;
            padding: 25px 30px;
            border: 1px solid #d4c5b2;
            background: #fcf8f0;
            border-radius: 2px;
            display: none;
            animation: fadeIn 0.6s ease;
        }
        
        .result-section.show {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-article-title {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.5em;
            font-weight: 700;
            color: #2c1810;
            text-align: center;
            border-bottom: 1px solid #d4c5b2;
            padding-bottom: 12px;
            margin-bottom: 15px;
        }
        
        .score-display {
            text-align: center;
            padding: 15px 0;
        }
        
        .score-number {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 4.5em;
            font-weight: 900;
            color: #2c1810;
            line-height: 1;
        }
        
        .score-out-of {
            font-family: 'Special Elite', cursive;
            font-size: 1em;
            color: #6b4c3b;
        }
        
        .score-bar {
            width: 100%;
            height: 6px;
            background: #e8dcc8;
            margin: 12px 0 20px 0;
            border-radius: 3px;
            overflow: hidden;
        }
        
        .score-bar-fill {
            height: 100%;
            background: #2c1810;
            transition: width 0.8s ease;
            border-radius: 3px;
        }
        
        .quality-label {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.6em;
            font-weight: 700;
            text-align: center;
            margin: 10px 0;
        }
        
        .quality-label.high { color: #2d6a4f; }
        .quality-label.medium { color: #b9770e; }
        .quality-label.low { color: #a4332a; }
        
        .feedback-box {
            margin: 15px 0 10px 0;
            padding: 15px 20px;
            background: #f5ede1;
            border-left: 4px solid #2c1810;
            font-family: 'Special Elite', cursive;
            font-size: 0.95em;
            color: #3d2b1f;
            line-height: 1.6;
        }
        
        .feedback-box strong {
            font-family: 'Playfair Display', Georgia, serif;
            font-weight: 700;
        }
        
        /* QUICK TIPS - NEWSPAPER SIDEBAR STYLE */
        .tips-section {
            margin-top: 30px;
            padding: 20px 25px;
            background: #f5ede1;
            border: 1px solid #d4c5b2;
            border-left: 4px solid #2c1810;
        }
        
        .tips-title {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1em;
            font-weight: 700;
            color: #2c1810;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        
        .tips-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px 25px;
            font-size: 0.85em;
            color: #3d2b1f;
        }
        
        .tips-grid li {
            list-style: none;
            padding: 4px 0;
            font-family: 'Source Serif Pro', Georgia, serif;
        }
        
        .tips-grid li::before {
            content: '◆ ';
            color: #8b6b5a;
            font-size: 0.7em;
        }
        
        .footer-note {
            text-align: center;
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #d4c5b2;
            font-family: 'Special Elite', cursive;
            font-size: 0.75em;
            color: #8b6b5a;
            letter-spacing: 2px;
        }
        
        /* RESPONSIVE */
        @media (max-width: 600px) {
            .newspaper-container { padding: 20px; }
            .publication-name { font-size: 2em; letter-spacing: 4px; }
            .headline { font-size: 1.6em; }
            .score-number { font-size: 3em; }
            .tips-grid { grid-template-columns: 1fr; }
            .edition-line { flex-direction: column; text-align: center; gap: 5px; }
        }
    </style>
</head>
<body>
    <div class="newspaper-container">
        
        <!-- HEADER -->
        <div class="newspaper-header">
            <div class="publication-name">The Essay Times</div>
            <div class="publication-subtitle">✦ Automated Scoring Edition ✦</div>
            <div class="edition-line">
                <span>Vol. I — No. 07</span>
                <span>{{ date }}</span>
                <span>Student Edition</span>
            </div>
        </div>
        
        <!-- HEADLINE -->
        <h1 class="headline">Write Better Essays.</h1>
        <p class="headline-sub">Smarter Writing. Stronger Impact.</p>
        
        <hr class="divider">
        
        <!-- ESSAY INPUT -->
        <div class="essay-section">
            <div class="section-title">Submit Your Essay</div>
            <form method="POST">
                <textarea name="essay" placeholder="Paste your essay here... Let our AI analyze your writing style, structure, and vocabulary.">{{ essay_text if essay_text else '' }}</textarea>
                <div class="button-container">
                    <button type="submit" class="btn-score">✦ Score My Essay ✦</button>
                </div>
            </form>
        </div>
        
        <hr class="divider-dotted">
        
        <!-- RESULTS -->
        {% if result %}
        <div class="result-section show">
            <div class="result-article-title">✧ Analysis Report ✧</div>
            
            <div class="score-display">
                <div class="score-number">{{ "%.1f"|format(result.score) }}<span class="score-out-of"> / 12</span></div>
                <div class="score-bar">
                    <div class="score-bar-fill" style="width: {{ (result.score / 12 * 100)|round }}%;"></div>
                </div>
                <div class="quality-label {{ result.class }}">{{ result.quality }}</div>
            </div>
            
            <div class="feedback-box">
                <strong>✦ Editor's Note:</strong> {{ result.feedback }}
            </div>
        </div>
        {% endif %}
        
        <!-- QUICK TIPS -->
        <div class="tips-section">
            <div class="tips-title">✧ Quick Tips for Better Essays</div>
            <ul class="tips-grid">
                <li>Be specific — concrete details make your points powerful</li>
                <li>Use evidence — back up ideas with facts and examples</li>
                <li>Stay focused — stick to your thesis and avoid going off track</li>
                <li>Edit ruthlessly — good writing is great rewriting</li>
            </ul>
        </div>
        
        <div class="footer-note">
            <span>✦ Well written is well thought. — William Zinsser ✦</span>
        </div>
        
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    essay_text = ""
    current_date = datetime.now().strftime('%B %d, %Y')
    
    if request.method == 'POST':
        essay_text = request.form.get('essay', '')
        if essay_text.strip():
            # Get prediction
            score = score_essay(essay_text)
            
            # Determine quality and feedback
            if score >= 8:
                quality = "✦ High Quality Essay ✦"
                feedback = "Excellent work! Your essay demonstrates strong arguments, a clear structure, and sophisticated vocabulary. The writing flows naturally and engages the reader effectively. Keep up this outstanding standard!"
                css_class = "high"
            elif score >= 5:
                quality = "✦ Medium Quality Essay ✦"
                feedback = "Good effort! Your essay has a solid foundation. To improve, consider adding more specific examples, strengthening your transitions between paragraphs, and expanding your vocabulary. You're on the right track!"
                css_class = "medium"
            else:
                quality = "✦ Low Quality Essay ✦"
                feedback = "This essay needs development. Focus on organizing your ideas more clearly, providing specific evidence for your arguments, and expanding your vocabulary. Remember: every great writer started somewhere — keep practicing!"
                css_class = "low"
            
            result = {
                'score': score,
                'quality': quality,
                'feedback': feedback,
                'class': css_class
            }
    
    return render_template_string(HTML_TEMPLATE, result=result, essay_text=essay_text, date=current_date)

if __name__ == '__main__':
    print("\n📰 The Essay Times - Automated Scoring System")
    print("=" * 50)
    print("🌐 Server running at: http://127.0.0.1:5000")
    print("📱 Open your browser and start scoring essays!")
    print("🔄 Press CTRL+C to stop")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)