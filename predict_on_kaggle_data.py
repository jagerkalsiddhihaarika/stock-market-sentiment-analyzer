import pandas as pd
from sentiment_analyzer import clean_text, vectorizer, model, apply_confidence_threshold

# Step 1: Load the Kaggle dataset
df = pd.read_csv("djia_news copy.csv")

# Step 2: Print available columns to confirm structure
print("🧾 Available Columns:", df.columns.tolist())
print(df.head())

# Step 3: Clean the text column (assuming it’s called 'headline')
df['cleaned_text'] = df['Headline'].astype(str).apply(clean_text)

# Step 4: Vectorize using your existing TF-IDF vectorizer
X = vectorizer.transform(df['cleaned_text'])

# Step 5: Predict using your trained model
predictions = model.predict(X)
confidences = model.predict_proba(X).max(axis=1) * 100

# Step 6: Apply confidence thresholding
final_preds = [apply_confidence_threshold(pred, conf)
               for pred, conf in zip(predictions, confidences)]

# Step 7: Add predictions to the dataframe
df['predicted_sentiment'] = final_preds
df['confidence (%)'] = confidences.round(2)

# Optional: Map existing numeric sentiment labels
if 'sentiment' in df.columns:
    df['true_label'] = df['sentiment'].map({0: 'negative', 1: 'positive', 2: 'neutral'})

# Step 8: Save to new CSV
df.to_csv("djia_labeled_by_model.csv", index=False)
print("✅ Output saved to djia_labeled_by_model.csv")
