import pandas as pd
import numpy as np
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import RandomOverSampler

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print("🚀 Script started!")

# Load dataset
data = pd.read_csv("all-data.csv", encoding='ISO-8859-1', header=None)
data.columns = ["sentiment", "text"]
data = data[data['sentiment'].isin(['positive', 'negative', 'neutral'])]

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = stop_words.union({"said", "would", "also", "could"})
negation_words = {"not", "no", "never", "n't", "cannot", "hardly", "barely", "doesn’t", "wasn’t"}

keyword_boosts = {
    # Negative
    "bad": "verybad", "worst": "verybad", "awful": "verybad", "hate": "verybad", "disappointed": "verybad",
    "poor": "verybad", "terrible": "verybad", "slow": "verybad", "crashes": "verybad", "useless": "verybad", "shit": "verybad",
    # Positive
    "love": "verygood", "great": "verygood", "amazing": "verygood", "good": "verygood", "happy": "verygood",
    "fantastic": "verygood", "excellent": "verygood", "delightful": "verygood", "awesome": "verygood",
    # Neutral
    "okay": "neutral", "fine": "neutral", "average": "neutral", "mediocre": "neutral"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", '', text)
    tokens = word_tokenize(text)

    new_tokens = []
    negate = False
    for token in tokens:
        if token in negation_words:
            negate = True
            continue

        word = keyword_boosts.get(token, token)

        if negate:
            if word == "verygood":
                word = "verybad"
            elif word == "verybad":
                word = "verygood"
            else:
                word = "not_" + word
            negate = False

        if word not in custom_stopwords and len(word) > 2:
            new_tokens.append(lemmatizer.lemmatize(word))

    return " ".join(new_tokens)

# Clean the dataset
data['cleaned_text'] = data['text'].apply(clean_text)
print("✅ Text cleaned.")

# Vectorize the text
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['sentiment']

# Balance the dataset
ros = RandomOverSampler(sampling_strategy='not majority', random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
print("🔁 Data balanced with oversampling.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print("🧠 Data split.")

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)
print("🤖 Model trained.")

# Evaluate the model
y_pred = model.predict(X_test)
print("\n📊 Evaluation Results:")
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, "stock_sentiment_nb_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("💾 Trained model and vectorizer saved successfully.")

# Emoji output testing loop
sentiment_emojis = {
    "positive": "😊",
    "neutral": "😐",
    "negative": "😠"
}

print("\n🧪 Enter news headlines to test. Type 'exit' to quit.")
while True:
    user_input = input("📰 Enter a headline: ")
    if user_input.lower() == "exit":
        print("👋 Exiting. Thank you!")
        break
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    confidence = np.max(model.predict_proba(vector)) * 100
    print(f"📈 Predicted Sentiment: {prediction.capitalize()} {sentiment_emojis[prediction]} (Confidence: {confidence:.2f}%)")
