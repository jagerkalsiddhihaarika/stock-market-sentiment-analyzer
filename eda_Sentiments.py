import pandas as pd
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

# 📦 Download NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# 🔗 Load Dataset
url = 'https://raw.githubusercontent.com/isaaccs/sentiment-analysis-for-financial-news/master/all-data.csv'
data = pd.read_csv(url, encoding="cp1252", header=None)
data.columns = ['sentiment', 'text']

# 📊 Basic Stats
def classical_data_stat(data, y):
    print(f'\n🔢 {data.shape[0]} observations, {data.shape[1]} columns')
    if not data.isnull().values.any():
        print("✅ No missing values")
    else:
        print("❌ Missing values found")
    if len(data[y].unique()) < 50:
        print(f"📌 '{y}' is a categorical feature")
        print("\nLabel Distribution:")
        print(data[y].value_counts())
        print("\nLabel Distribution (%):")
        print(100 * data[y].value_counts() / data.shape[0])
    else:
        print(f"📌 '{y}' looks continuous")

classical_data_stat(data, 'sentiment')

# 🧹 Text Preprocessing
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(sentence):
    sentence = str(sentence).lower()
    sentence = re.sub('<.*?>', '', sentence)
    sentence = re.sub(r'http\S+', '', sentence)
    sentence = re.sub('[0-9]+', '', sentence)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in tokens if w not in stop_words and len(w) > 2]
    stemmed = [stemmer.stem(w) for w in filtered_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in stemmed]
    return " ".join(lemmatized)

data['text'] = data['text'].apply(preprocess)

# 🔠 N-grams
def ngrams(input_list, n):
    return ['_'.join(input_list[i:i+n]) for i in range(len(input_list)-n+1)]

# Filter for known labels
tags = ['neutral', 'negative', 'positive']
data_bis = data[data.sentiment.isin(tags)].copy()

data_bis['Tokens'] = data_bis['text'].apply(nltk.word_tokenize)
data_bis['bi_Grams'] = data_bis['Tokens'].apply(lambda x: ngrams(x, 2))
data_bis['tri_Grams'] = data_bis['Tokens'].apply(lambda x: ngrams(x, 3))

print(f"\n✅ {len(data_bis)} samples across {len(tags)} classes")

# ☁️ Word Cloud
def plot_word_cloud(data, text='text', label=None):
    word_cloud_data = " ".join(data[text])
    cloud = WordCloud(stopwords=STOPWORDS).generate(word_cloud_data)
    plt.figure(figsize=(8, 6))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Full Corpus WordCloud")
    plt.tight_layout()
    plt.savefig("wordcloud_all.png")
    plt.close()

    if label:
        for lbl in data[label].unique():
            subset = data[data[label] == lbl]
            cloud = WordCloud(stopwords=STOPWORDS).generate(" ".join(subset[text]))
            plt.figure()
            plt.imshow(cloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"WordCloud for {lbl}")
            plt.tight_layout()
            plt.savefig(f"wordcloud_{lbl}.png")
            plt.close()

    print("✅ WordClouds saved as images in your folder.")

plot_word_cloud(data_bis, text='text', label='sentiment')

# 📊 Class-specific word stats
def stat_des_text(data, col, text, word_col):
    labels = data[col].unique()
    tag_dict = {tag: Counter() for tag in labels}
    global_counter = Counter()

    for i in data.index:
        tag = data.loc[i][col]
        tag_dict[tag].update(set(data.loc[i][word_col]))
        global_counter.update(set(data.loc[i][word_col]))

    for stop_count in [50, 500, 1000]:
        print(f"\n🛑 Ignoring top {stop_count} most common words")
        StopWords = [word for word, _ in global_counter.most_common(stop_count)]

        for tag in labels:
            print(f"\n📍 Top words for '{tag}' class:")
            tag_len = len(data[data[col] == tag])
            count = 0
            for word, freq in tag_dict[tag].most_common(500):
                if word not in StopWords:
                    percent = round(100 * freq / tag_len, 2)
                    print(f"{word:<20} {percent:>5}%")
                    count += 1
                if count == 15:
                    break

stat_des_text(data_bis, 'sentiment', 'text', 'bi_Grams')

