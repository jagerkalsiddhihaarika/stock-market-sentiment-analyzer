# plots.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import os

# Load CSV
df = pd.read_csv("djia_labeled_by_model.csv")

# Normalize column names
df.columns = [col.lower().replace(" ", "_") for col in df.columns]
if 'predicted_label' not in df.columns and 'predicted_sentiment' in df.columns:
    df['predicted_label'] = df['predicted_sentiment']
if 'predicted_label' not in df.columns:
    df['predicted_label'] = np.random.choice(['positive', 'neutral', 'negative'], len(df))

# Handle missing ticker
if 'ticker' not in df.columns:
    df['ticker'] = np.random.choice(['AAPL', 'GOOG', 'MSFT', 'TSLA'], len(df))

# Handle missing confidence
if 'confidence' not in df.columns:
    df['confidence'] = np.random.uniform(0.5, 1.0, len(df))

# Simulate prices if not present
if 'price' not in df.columns:
    df['price'] = np.random.normal(150, 20, len(df))

# Create directory
os.makedirs("plots", exist_ok=True)

# 1. Sentiment Distribution by Ticker
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='ticker', hue='predicted_label')
plt.title("Sentiment Distribution by Ticker")
plt.savefig("plots/sentiment_distribution_by_ticker.png")
plt.close()

# 2. Word Clouds
for sentiment in ['positive', 'negative', 'neutral']:
    text = ' '.join(df[df['predicted_label'] == sentiment]['headline'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud.to_file(f"plots/wordcloud_{sentiment}.png")

# Word cloud for all headlines
all_text = ' '.join(df['headline'].dropna().astype(str))
wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate(all_text)
wordcloud_all.to_file("plots/wordcloud_all.png")

# 3. Simulated Sentiment Trend
if 'date' not in df.columns:
    df['date'] = pd.date_range("2023-01-01", periods=len(df), freq='D')
else:
    df['date'] = pd.to_datetime(df['date'])

sentiment_trend = df.groupby(['date', 'predicted_label']).size().unstack(fill_value=0)
sentiment_trend = sentiment_trend.sort_index()
sentiment_trend.rolling(30).mean().plot(figsize=(14, 6), title="Simulated Sentiment Trend (30-day rolling avg)")
plt.ylabel("Headline Count")
plt.savefig("plots/simulated_sentiment_trend.png")
plt.close()

# 4. Bar Plot: Overall Sentiment Counts
plt.figure(figsize=(8, 5))
sns.countplot(x='predicted_label', data=df, order=['positive', 'neutral', 'negative'])
plt.title("Overall Sentiment Count")
plt.savefig("plots/barplot_sentiment_counts.png")
plt.close()

# 5. Sentiment vs. Ticker Heatmap
heatmap_data = pd.crosstab(df['ticker'], df['predicted_label'])
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Heatmap: Sentiment vs. Ticker")
plt.savefig("plots/heatmap_sentiment_vs_ticker.png")
plt.close()

# 6. Sentiment vs. Stock Price Correlation
plt.figure(figsize=(10, 6))
sns.boxplot(x='predicted_label', y='price', data=df)
plt.title("Sentiment vs. Stock Price (Simulated)")
plt.savefig("plots/sentiment_vs_stock_price.png")
plt.close()

# 7. Confidence by Company
plt.figure(figsize=(10, 6))
sns.boxplot(x='ticker', y='confidence', data=df)
plt.title("Model Confidence by Company")
plt.savefig("plots/confidence_by_company.png")
plt.close()

# 8. Pie Chart of Sentiment Proportions
plt.figure(figsize=(6, 6))
df['predicted_label'].value_counts().plot.pie(autopct='%1.1f%%', colors=['green', 'gray', 'red'])
plt.title("Sentiment Proportions")
plt.ylabel("")
plt.savefig("plots/sentiment_pie_chart.png")
plt.close()

print("✅ All plots saved in 'plots/' folder.")
