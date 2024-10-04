!pip install beautifulsoup4 requests transformers torch textblob

import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
from textblob import TextBlob

# دالة لسحب البيانات من موقع معين
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # استخراج جميع الفقرات
    paragraphs = soup.find_all('p')
    texts = [para.get_text() for para in paragraphs]
    
    return texts

# cnn  استخدام الدالة لسحب البيانات من موقع
url = 'https://www.cnn.com'  # قم بتغيير هذا الرابط إلى الرابط الذي تود سحب البيانات منه
docs = scrape_website(url)

# تهيئة نموذج BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# دالة لتحويل النص إلى تمثيل عددي باستخدام BERT
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# دالة لتحليل المشاعر
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # إرجاع قيمة المشاعر

# دالة لحساب وعرض المقالات المتشابهة
def get_similar_articles(query, docs):
    print("Query:", query)
    print("Top articles with highest cosine similarity:")
    
    query_embedding = get_embeddings(query)
    
    sim = {}
    for i, doc in enumerate(docs):
        doc_embedding = get_embeddings(doc)
        cosine_similarity = torch.cosine_similarity(query_embedding, doc_embedding)
        sim[i] = cosine_similarity.item()
    
    # فرز النتائج
    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    
    for index, value in sim_sorted:
        if value > 0.0:
            sentiment_score = analyze_sentiment(docs[index])
            print(f"Similarity Score: {value:.4f} - Article: {docs[index]}")
            print(f"Sentiment Score: {sentiment_score:.4f}")
            print()

# مثال لاستعلامات متعددة
queries = ['Google search engine', 'smartphone technology', 'Snapchat messaging']

for q in queries:
    get_similar_articles(q, docs)
    print('-' * 100)

النتائج 
Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (4.12.3)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (2.32.3)
Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.44.2)
Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (2.4.1+cu121)
Requirement already satisfied: textblob in /usr/local/lib/python3.10/dist-packages (0.17.1)
Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4) (2.6)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests) (2024.8.30)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.16.1)
Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.24.7)
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (1.26.4)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (24.1)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.2)
Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2024.9.11)
Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.5)
Requirement already satisfied: tokenizers<0.20,>=0.19 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.19.1)
Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.5)
Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch) (4.12.2)
Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch) (1.13.3)
Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch) (3.3)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch) (3.1.4)
Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch) (2024.6.1)
Requirement already satisfied: nltk>=3.1 in /usr/local/lib/python3.10/dist-packages (from textblob) (3.8.1)
Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk>=3.1->textblob) (8.1.7)
Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk>=3.1->textblob) (1.4.2)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (2.1.5)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->torch) (1.3.0)
Query: Google search engine
Top articles with highest cosine similarity:
Similarity Score: 0.4920 - Article: Show all
Sentiment Score: 0.0000

Similarity Score: 0.4920 - Article: Show all
Sentiment Score: 0.0000

Similarity Score: 0.4920 - Article: Show all
Sentiment Score: 0.0000

Similarity Score: 0.4696 - Article: © 2024 Cable News Network. A Warner Bros. Discovery Company. All Rights Reserved.  CNN Sans ™ & © 2016 Cable News Network.
Sentiment Score: 0.0000

----------------------------------------------------------------------------------------------------
Query: smartphone technology
Top articles with highest cosine similarity:
Similarity Score: 0.6374 - Article: Show all
Sentiment Score: 0.0000

Similarity Score: 0.6374 - Article: Show all
Sentiment Score: 0.0000

Similarity Score: 0.6374 - Article: Show all
Sentiment Score: 0.0000

Similarity Score: 0.4403 - Article: © 2024 Cable News Network. A Warner Bros. Discovery Company. All Rights Reserved.  CNN Sans ™ & © 2016 Cable News Network.
Sentiment Score: 0.0000

----------------------------------------------------------------------------------------------------
Query: Snapchat messaging
Top articles with highest cosine similarity:
Similarity Score: 0.5114 - Article: © 2024 Cable News Network. A Warner Bros. Discovery Company. All Rights Reserved.  CNN Sans ™ & © 2016 Cable News Network.
Sentiment Score: 0.0000

Similarity Score: 0.5113 - Article: Show all
Sentiment Score: 0.0000

Similarity Score: 0.5113 - Article: Show all
Sentiment Score: 0.0000

Similarity Score: 0.5113 - Article: Show all
Sentiment Score: 0.0000

----------------------------------------------------------------------------------------------------

