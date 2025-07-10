import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import RandomOverSampler

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print("🚀 Script started!")

# Load dataset
data = pd.read_csv("all-data.csv", encoding='ISO-8859-1', header=None)
data.columns = ["sentiment", "text"]
data = data[data['sentiment'].isin(['positive', 'negative', 'neutral'])]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = stop_words.union({"said", "would", "also", "could"})
negation_words = {"not", "no", "never", "n't", "cannot", "hardly", "barely", "doesn’t", "wasn’t"}

# Strong Sentiment Sets
strong_negative = {
    "hate", "crash", "crashes", "crashed", "uninstall", "shit", "useless", "terrible", "worst",
    "frustrating", "buggy", "horrible", "disappointed", "sucks", "ruined", "angry", "annoyed", "bad",
    "cry", "slow", "disaster", "broken", "pathetic", "laggy", "awful", "garbage", "painful", "can't use",
    "incompetent", "problematic", "unstable", "fail", "failed", "failing", "regret", "not worth",
    "cancel my order", "refund", "never again", "hopeless", "crashing constantly", "bug", "bugs", "waste"
}

strong_positive = {
    "love", "amazing", "awesome", "fantastic", "best", "great", "happy", "delightful", "smooth",
    "excellent", "appreciate", "enjoy", "like", "superb", "wow", "brilliant", "lovely", "cool",
    "impressive", "clean", "perfect", "works well", "user-friendly", "recommend", "nice", "super", 
    "pleased", "satisfied", "better than expected", "top-notch", "marvelous", "wonderful", "5 stars",
    "excited", "blown away", "no issues", "stable", "good update", "useful", "reliable",
    "worth it", "worth every penny", "love this app", "works like charm", "top quality", "game changer"
}

neutral_words = {
    "okay", "fine", "meh", "average", "alright", "mediocre", "decent", "normal", "regular", "nothing",
    "typical", "common", "standard", "expected", "just", "ok", "not bad", "fair", "usual", "basic",
    "so-so", "kind of", "somewhat", "ordinary", "not special", "just okay", "not great", "not terrible",
    "passable", "it is what it is", "acceptable", "standard update", "no big deal"
}

# Text cleaning function
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

        word = lemmatizer.lemmatize(token)

        if negate:
            if word in strong_positive:
                word = "verybad"
            elif word in strong_negative:
                word = "verygood"
            else:
                word = "not_" + word
            negate = False
        elif word in strong_positive:
            word = "verygood"
        elif word in strong_negative:
            word = "verybad"
        elif word in neutral_words:
            word = "neutral"

        if word not in custom_stopwords and len(word) > 2:
            new_tokens.append(word)

    return " ".join(new_tokens)

# Clean the data
data['cleaned_text'] = data['text'].apply(clean_text)
print("✅ Text cleaned.")

# Vectorize
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['sentiment']

# Oversample
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
print("🔁 Data balanced with oversampling.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print("🧠 Data split.")

# Train classifier
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
print("🤖 Model trained.")

# Evaluation
y_pred = model.predict(X_test)
print("\n📊 Evaluation Results:")
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Sentiment display
sentiment_emojis = {
    "positive": "😊",
    "neutral": "😐",
    "negative": "😠",
    "uncertain": "❓"
}

def apply_confidence_threshold(pred_label, confidence):
    if pred_label == "neutral" and confidence < 65:
        return "uncertain"
    return pred_label

# Interactive prediction loop
print("\n🧪 Enter news headlines to test. Type 'exit' to quit.")

sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0, "uncertain": 0}

while True:
    user_input = input("📰 Enter a headline: ")
    if user_input.lower().strip() == "exit":
        print("\n📌 Final Verdict based on your inputs:")
        for sentiment, count in sentiment_counts.items():
            print(f"🔢 {sentiment.capitalize()}: {count}")
        # Company Verdict Logic
        pos = sentiment_counts["positive"]
        neg = sentiment_counts["negative"]
        total = sum(sentiment_counts.values()) - sentiment_counts["uncertain"]

        if total == 0:
            print("➖ Verdict: Not enough data to evaluate.")
        elif pos / total >= 0.6:
            print("✅ Verdict: Mostly Positive. Company is doing well!")
        elif neg / total >= 0.5:
            print("❌ Verdict: Mostly Negative. Company has issues.")
        else:
            print("➖ Verdict: Mixed feedback. Needs improvement or more reviews.")
        print("👋 Exiting. Thank you!")
        break

    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    confidence = np.max(model.predict_proba(vector)) * 100
    final_prediction = apply_confidence_threshold(prediction, confidence)

    sentiment_counts[final_prediction] += 1
    print(f"📈 Predicted Sentiment: {final_prediction.capitalize()} {sentiment_emojis[final_prediction]} (Confidence: {confidence:.2f}%)")
import pickle

# Save the trained model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
