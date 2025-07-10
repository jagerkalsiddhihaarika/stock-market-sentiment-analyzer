import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Start
print("🚀 Script started!")

# Download stopwords
nltk.download('stopwords')

# Load CSV
try:
    data = pd.read_csv("all-data.csv", encoding='ISO-8859-1', header=None)
    data.columns = ['Sentiment', 'Headline']
    print("✅ Data loaded.")
except Exception as e:
    print(f"❌ Could not load CSV: {e}")
    exit()

# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    return ' '.join([w for w in words if w not in stop_words])

data['Cleaned_Headline'] = data['Headline'].apply(clean_text)
y = data['Sentiment'].str.lower()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)  # includes bigrams
X = vectorizer.fit_transform(data['Cleaned_Headline']).toarray()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Oversample to balance classes
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
print("🔁 Data balanced with oversampling.")

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)
print("🤖 Model trained.")

# Evaluate
y_pred = model.predict(X_test)
print("📊 Evaluation Results:")
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📋 Classification Report:\n", classification_report(y_test, y_pred))

# Predict loop
print("\n🧪 Enter news headlines to test. Type 'exit' to quit.")
sentiment_map = {
    'negative': 'Negative 😟',
    'neutral': 'Neutral 😐',
    'positive': 'Positive 😊'
}

while True:
    headline = input("📰 Enter a headline: ")
    if headline.lower() == 'exit':
        print("👋 Exiting. Thank you!")
        break
    cleaned = clean_text(headline)
    vector = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(vector)[0]
    print("📈 Predicted Sentiment:", sentiment_map.get(pred, pred))
