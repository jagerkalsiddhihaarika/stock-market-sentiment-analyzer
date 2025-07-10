# 📊 Stock Market Sentiment Analyzer

![Sentiment Dashboard](plots/sentiment_distribution_by_ticker.png)

A machine learning-based tool that analyzes public sentiment from stock-related news headlines and classifies them as **positive**, **negative**, **neutral**, or **uncertain**. The tool also visualizes sentiment trends and helps investors evaluate market perception around companies.

---

## 🚀 Features

- 🧠 **Sentiment Classification** using TF-IDF + Logistic Regression  
- 📉 **Handles Class Imbalance** using RandomOverSampler  
- 🛠️ **Keyword Boosting + Negation Handling**  
- 📈 **Data Visualizations** (Time series, Ticker-wise, WordClouds, Heatmaps)  
- 🖥️ **Terminal-Based Sentiment UI**  
- 🧾 **Headline Confidence Scoring**  
- 📊 **Sentiment Trends vs Stock Tickers**

---

## 🧠 Model & Methodology

| Step | Description |
|------|-------------|
| **1. Preprocessing** | Lowercasing, removing punctuation, stopword removal, lemmatization |
| **2. Keyword Boosting** | Custom weights for words like `not good`, `excellent`, etc. |
| **3. Classification** | Logistic Regression with TF-IDF features |
| **4. Class Balancing** | RandomOverSampler for unbiased training |
| **5. Confidence Scoring** | Softmax-based score thresholding |

📈 **Accuracy Achieved:** `87.09%`

---

## 🗂️ Dataset

📄 **Kaggle Dataset Used:**  
[`djia_news copy.csv`](https://www.kaggle.com/datasets/sbhatti/stock-news-dataset)  
> Historical headlines with stock tickers from Dow Jones Industrial Average (DJIA)

🧾 Sample Columns:
- `Label`: Ground truth sentiment
- `Ticker`: Company symbol (e.g., AAPL, MSFT)
- `Headline`: News title
- `cleaned_text`: Preprocessed headline
- `predicted_sentiment`: Model prediction
- `confidence (%)`: Sentiment confidence

---

## 📊 Visualizations

### 🔵 Sentiment Distribution by Ticker
![Ticker Sentiment](plots/sentiment_distribution_by_ticker.png)

---

### 📍 WordClouds
| Positive 😊 | Neutral 😐 | Negative 😠 |
|------------|------------|-------------|
| ![Pos](plots/wordcloud_positive.png) | ![Neu](plots/wordcloud_neutral.png) | ![Neg](plots/wordcloud_negative.png) |

---

### 📈 Simulated Sentiment Trend
![Trend](plots/simulated_sentiment_trend.png)

---

### 🔥 Sentiment-Stock Correlation (Simulated)
![Correlation](plots/sentiment_vs_stock_price.png)

---

### 📊 Sentiment Heatmap by Company
![Heatmap](plots/heatmap_sentiment_vs_ticker.png)

---

## 🧪 How to Use

```bash
# Step 1: Install required packages
pip install -r requirements.txt

# Step 2: Run sentiment classifier and interact with UI
python sentiment_analyzer.py

# Step 3: Predict sentiments on Kaggle dataset
python predict_on_kaggle_data.py

# Step 4: See full dashboard of plots
python plots.py
