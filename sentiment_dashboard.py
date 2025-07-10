
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# 🔄 Load the labeled dataset
print("🔄 Loading dataset...")
df = pd.read_csv("djia_labeled_by_model.csv")

# 🧹 Basic Cleaning
df['date'] = pd.to_datetime(df['Date'], errors='coerce')
df['month'] = df['date'].dt.to_period('M')
df['sentiment'] = df['Predicted'].map({0: 'Negative', 1: 'Positive', 2: 'Neutral', 3: 'Uncertain'})
df['Label'] = df['Label'].map({0: 'Stock Down', 1: 'Stock Up', 2: 'Stock Flat'})

# 📊 1. Sentiment Distribution by Company (Bar Chart)
plt.figure(figsize=(14, 7))
sns.countplot(data=df, x='Ticker', hue='sentiment', palette='coolwarm')
plt.title("🏢 Sentiment Distribution per Company")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("sentiment_distribution_by_ticker.png")
plt.show()

# 📈 2. Monthly Sentiment Trends (Line Plot)
monthly_counts = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
monthly_counts.plot(kind='line', figsize=(12, 6))
plt.title("📅 Monthly Sentiment Trend Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Headlines")
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_sentiment_trend.png")
plt.show()

# 📉 3. Sentiment vs Stock Movement (Heatmap)
heatmap_data = pd.crosstab(df['sentiment'], df['Label'])
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='coolwarm')
plt.title("🔥 Sentiment Prediction vs. Actual Stock Movement")
plt.xlabel("Actual Stock Movement")
plt.ylabel("Predicted Sentiment")
plt.tight_layout()
plt.savefig("sentiment_vs_stock_movement_heatmap.png")
plt.show()

# 💨 4. Word Clouds by Sentiment
def generate_wordcloud(df_subset, label, file_name):
    text = " ".join(str(h) for h in df_subset['Headline'].dropna())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{label} Word Cloud")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

# Generate for each sentiment
generate_wordcloud(df[df['sentiment'] == 'Positive'], "Positive 😊", "wordcloud_positive.png")
generate_wordcloud(df[df['sentiment'] == 'Negative'], "Negative 😠", "wordcloud_negative.png")
generate_wordcloud(df[df['sentiment'] == 'Neutral'], "Neutral 😐", "wordcloud_neutral.png")
generate_wordcloud(df, "All Sentiments 🌍", "wordcloud_all.png")

print("✅ All visualizations generated and saved.")
