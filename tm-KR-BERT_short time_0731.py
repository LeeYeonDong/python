import pandas as pd
from bertopic import BERTopic
from transformers import BertTokenizer, BertModel
from transformers import BertTokenizerFast, AlbertModel
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch
import re
import time
from umap import UMAP
from hdbscan import HDBSCAN
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
from konlpy.tag import Okt
from itertools import product
from gensim import matutils
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import itertools
import ast

# 데이터 로드
df = pd.read_csv('D:/대학원/논문/소논문/부동산_토픽모델링/부동산_수정_df.csv', encoding='utf-8')
#df = df[:10000]

# 중복된 "제목" 제거 및 고유 ID 부여
df = df.drop_duplicates(subset=['제목']).reset_index(drop=True)
df['id'] = df.index + 1

# 날짜 및 시간 문자열 변환 함수
def convert_korean_datetime(dt_str):
    dt_str = dt_str.replace('오후', 'PM').replace('오전', 'AM')
    return pd.to_datetime(dt_str, format='%Y.%m.%d. %p %I:%M')

# 날짜 칼럼 변환
df['날짜'] = df['날짜'].apply(convert_korean_datetime)
df = df[(df['날짜'].dt.year >= 2012) & (df['날짜'].dt.year <= 2022)]

# 특수문자로 시작하는 단어 제거
df['제목'] = df['제목'].str.replace(r'\[[^\]]*\]', '', regex=True)
df['제목'] = df['제목'].str.replace(r'<[^>]*>', '', regex=True)
df['제목'] = df['제목'].str.replace(r'\{[^\}]*\}', '', regex=True)
df['제목'] = df['제목'].str.replace(r'@[^\s]+', '', regex=True)
df['제목'] = df['제목'].str.replace(r'[^\w\s,.?!\'"]', '', regex=True)
df['제목'] = df['제목'].str.replace(r'\b\w\b', '', regex=True)
df['제목'] = df['제목'].str.replace(r'\d+년|\d+월|\d+일', ' ', regex=True).str.strip()
df['제목'] = df['제목'].str.replace(r'들썩|벌써|쑥쑥|헤드라인|머니투데이', ' ', regex=True).str.strip()
df['제목'] = df['제목'].str.replace(r'뉴스|오늘|오프라인|종목', ' ', regex=True).str.strip()
df['제목'] = df['제목'].str.replace(r'전하|매경|우리은행|레이더|글쎄', ' ', regex=True).str.strip()
df['제목'] = df['제목'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Okt 형태소 분석기 인스턴스 생성
okt = Okt()

# 텍스트를 토큰화하는 함수
def tokenize(text):
    tokens = okt.morphs(text)
    return tokens

# 결측값을 빈 문자열로 대체하고, 모든 입력을 문자열로 변환
df['제목'] = df['제목'].fillna('').astype(str)

# '제목' 컬럼에 대해 토큰화
df['제목'] = df['제목'].apply(tokenize)

# 비문자열 데이터 타입 변환
df['제목'] = df['제목'].astype(str)

# 문서 데이터 준비
documents = df['제목'].tolist()

# tokenized_docs 정의
tokenized_docs = [ast.literal_eval(doc) for doc in documents]

# gensim 사전과 코퍼스 생성
dictionary = Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

# 문서 임베딩 함수 정의
def embed_documents(documents, model, tokenizer, device='cuda', batch_size=16):
    model.to(device)
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        inputs = tokenizer(batch_docs, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# KR-BERT 모델 초기화
tokenizer = BertTokenizer.from_pretrained('snunlp/KR-BERT-char16424')
model = BertModel.from_pretrained('snunlp/KR-BERT-char16424')

# 랜덤 시드 고정
np.random.seed(1029)
torch.manual_seed(1029)

# UMAP과 HDBSCAN 객체를 생성하고 BERTopic에 전달하는 방법
umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine')
hdbscan_model = HDBSCAN(min_cluster_size=200, min_samples=100, metric='manhattan', prediction_data=True)

# 토픽 모델링 및 Coherence Score 계산
coherence_scores = []
topic_counts = range(2, 20, 2)

for nr_topics in topic_counts:
    # BERTopic 모델 초기화 및 훈련
    topic_model = BERTopic(embedding_model=lambda docs: embed_documents(docs, model, tokenizer),
                           umap_model=umap_model,
                           hdbscan_model=hdbscan_model,
                           verbose=True,
                           calculate_probabilities=True,
                           nr_topics=nr_topics)

    topics, _ = topic_model.fit_transform(df['제목'].tolist())

    # 토픽별 키워드 추출
    topic_words = []
    for topic in set(topics):
        if topic != -1:
            topic_words.append([word for word, _ in topic_model.get_topic(topic)])

    # Coherence Score 계산
    coherence_model = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    coherence_scores.append(coherence_score)

# Coherence Score 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(topic_counts, coherence_scores, marker='o')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.title('Coherence Score by Number of Topics')
plt.show()

# 최적의 토픽 수 출력
optimal_topic_count = topic_counts[np.argmax(coherence_scores)]
print(f'Optimal number of topics: {optimal_topic_count}')