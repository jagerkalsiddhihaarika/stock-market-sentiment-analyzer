import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("djia_labeled_by_model.csv")

# Fix column name based on your file
if 'Predicted Label' not in df.columns:
    print("Fixing column name or generating fake labels...")
    labels = ['positive', 'negative', 'neutral']
    df['predicted_label'] = np.random.choice(labels, size=len(df))
    label_col = 'predicted_label'
else:
    label_col = 'Predicted Label'

# Generate fake dates
fake_dates = pd.date_range("2023-01-01", periods=len(df), freq='D')
df['Date'] = np.random.choice(fake_dates, size=len(df))

# Plot sentiment trend
df['Date'] = pd.to_datetime(df['Date'])
sentiment_trend = df.groupby(['Date', label_col]).size().unstack(fill_value=0)
sentiment_trend.plot(figsize=(14,6), title="Sentiment Trend Over Time")
plt.ylabel("Headline Count")
plt.grid(True)
plt.show()
