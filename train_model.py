# train_model.py - Regenerate model files
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import pickle

print("=" * 60)
print("🔧 TRAINING MODEL - Please Wait...")
print("=" * 60)

# 1. Load dataset
try:
    df = pd.read_csv('training_set_rel3.tsv', sep='\t', encoding='latin-1')
    print("✅ Dataset loaded!")
except FileNotFoundError:
    print("❌ training_set_rel3.tsv not found!")
    print("📁 Please make sure the dataset is in this folder:")
    import os
    print(f"   Current folder: {os.getcwd()}")
    exit()

# 2. Use 3000 essays
df = df.head(3000)
print(f"📊 Using {len(df)} essays")

# 3. Clean function
def clean_essay(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

print("🧹 Cleaning essays...")
df['cleaned_essay'] = df['essay'].apply(clean_essay)

# 4. TF-IDF
print("🔢 Converting text to numbers (TF-IDF)...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(df['cleaned_essay'])
y = df['domain1_score']

print(f"✅ Created {X.shape[1]} features from {X.shape[0]} essays")

# 5. Train model
print("🤖 Training Linear Regression model...")
model = LinearRegression()
model.fit(X, y)
print("✅ Model trained!")

# 6. Save files
print("💾 Saving model files...")

with open('essay_scorer_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ essay_scorer_model.pkl saved")

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ vectorizer.pkl saved")

# 7. Test the model
print("\n🧪 Testing the model...")
test_essay = "The internet has revolutionized education by making information accessible to everyone."
cleaned = clean_essay(test_essay)
features = vectorizer.transform([cleaned])
score = model.predict(features)[0]
print(f"📝 Test Essay: {test_essay}")
print(f"🎯 Score: {score:.1f}/12")

print("\n" + "=" * 60)
print("✅ DONE! Model files created successfully!")
print("🚀 Now run: python quick_app.py")
print("=" * 60)