
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import nltk
from nltk.corpus import stopwords
import pickle

# Download stopwords
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()



# LOAD DATA 
def load_dataset(filename):
    encodings = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
    for encoding in encodings:
        try:
            print(f" Loading with {encoding}...")
            df = pd.read_csv(filename, sep='\t', encoding=encoding)
            print(f" Loaded successfully!")
            return df
        except:
            continue
    raise Exception(" Could not load file!")

df = load_dataset('training_set_rel3.tsv')

# Use 1000 essays
df = df.head(1000)
print(f" Using {len(df)} essays for training")

# Clean function
def clean_essay(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

df['cleaned_essay'] = df['essay'].apply(clean_essay)

#  PREPARE FEATURES AND TARGET 
print("\n Preparing features...")
X = df['cleaned_essay']
y = df['domain1_score']

print(f" Number of essays: {len(X)}")
print(f" Score range: {y.min()} to {y.max()}")

# CONVERT TEXT TO NUMBERS (TF-IDF) 
print("\n Converting text to numbers (TF-IDF)...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2
)

X_features = vectorizer.fit_transform(X)
print(f" Created {X_features.shape[1]} numerical features from words")
print(f" Feature matrix size: {X_features.shape}")

#  SPLIT DATA 
print("\n Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.20, random_state=42
)

# FIXED: Use .shape[0] for sparse matrices
print(f" Training set: {X_train.shape[0]} essays")
print(f" Testing set: {X_test.shape[0]} essays")

#  TRAIN THE MODEL 
print("\n Training the scoring model...")

model = LinearRegression()
model.fit(X_train, y_train)

print(" Model training complete!")

# TEST THE MODEL 
print("\n Testing predictions...")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f" Model Performance:")
print(f"  • Mean Squared Error: {mse:.3f}")
print(f"  • R² Score: {r2:.3f}")
print(f"  • (R² of 1.0 = perfect, 0.0 = random guessing)")

#  SHOW SAMPLE PREDICTIONS 
print("\n Sample Predictions (First 5 test essays):")
print("-" * 60)
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    difference = abs(actual - predicted)
    print(f"Essay {i+1}:")
    print(f"  Actual Score: {actual:.0f}")
    print(f"  Predicted Score: {predicted:.1f}")
    print(f"  Difference: {difference:.1f}")
    print()

#  IMPORTANT WORDS 
print("\n Top 10 Most Important Words for Scoring:")
print("-" * 60)

feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_

important_words = sorted(
    zip(feature_names, coefs),
    key=lambda x: abs(x[1]),
    reverse=True
)

print("Positive words (increases score):")
for word, coef in important_words[:5]:
    if coef > 0:
        print(f"  +{coef:.3f}: '{word}'")

print("\nNegative words (decreases score):")
for word, coef in important_words[-5:]:
    if coef < 0:
        print(f"  {coef:.3f}: '{word}'")

#  SAVE MODEL 
print("\n Saving model...")

with open('essay_scorer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print(" Model saved as 'essay_scorer_model.pkl'")
print(" Vectorizer saved as 'vectorizer.pkl'")

#  TEST WITH YOUR OWN ESSAY 
print("\n SCORE YOUR OWN ESSAY:")
print("-" * 60)

def score_essay(essay_text):
    cleaned = clean_essay(essay_text)
    features = vectorizer.transform([cleaned])
    score = model.predict(features)[0]
    return score

# Try it with a sample essay
test_essay = """
Technology has revolutionized education in ways we never imagined. 
Students today have access to unlimited information at their fingertips. 
However, this also creates challenges in maintaining focus and academic integrity.
"""

predicted_score = score_essay(test_essay)
print(f" Test Essay:")
print(f"{test_essay}")
print(f"\n Predicted Score: {predicted_score:.1f} out of 12")

# Score range interpretation
if predicted_score > 8:
    quality = "High quality"
elif predicted_score > 5:
    quality = "Medium quality"
else:
    quality = "Low quality"
print(f"📊 This essay is rated as: {quality}")

print("\n" + "=" * 60)
print("🎉 DAY 2 COMPLETE! Your essay scoring system is working!")
print("📈 Next: Day 3 - Create a web interface")
