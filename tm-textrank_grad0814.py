import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from collections import Counter
import numpy as np
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
import stanza
from infomap import Infomap
from math import log2
import re
from random import randint, random
import math

# 파일 경로 설정
# file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_sample_processed.csv'
file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_processed.csv' # 수정

# CSV 파일 불러오기
df_filtered = pd.read_csv(file_path)

# keyword 열에서 가장 많은 키워드 수 계산

nltk.download('punkt')

#### 1. textrank
## abstract에서 keyword를 Textrank를 사용하여 추출
df_filtered_1 = df_filtered.copy()

# Textrank 키워드 추출 함수
def textrank_keywords(text, top_n=5):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = (X * X.T).toarray()
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    keywords = []
    for score, sentence in ranked_sentences:
        keywords.extend(word_tokenize(sentence.lower()))
        if len(set(keywords)) >= top_n:
            break
    
    return list(set(keywords))[:top_n]

# 추출된 키워드를 데이터 프레임에 추가
df_filtered_1['extracted_keywords'] = df_filtered_1['abstract'].apply(lambda x: textrank_keywords(x, top_n=5))

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_1[['abstract', 'keywords', 'extracted_keywords']])

# Precision, Recall, F1 계산 함수
def calculate_metrics(extracted, actual):
    extracted_set = set(extracted)
    actual_set = set(actual)
    
    true_positive = len(extracted_set & actual_set)
    false_positive = len(extracted_set - actual_set)
    false_negative = len(actual_set - extracted_set)
    
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

df_filtered_1['metrics'] = df_filtered_1.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)
df_filtered_1[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_1['metrics'].tolist(), index=df_filtered_1.index)

print(df_filtered_1[['keywords', 'extracted_keywords', 'precision', 'recall', 'f1']].head())

# ROUGE 값 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores


df_filtered_1['rouge'] = df_filtered_1.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_1['rouge1'] = df_filtered_1['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_1['rouge2'] = df_filtered_1['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_1['rougeL'] = df_filtered_1['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result1 = df_filtered_1[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

print(df_result1)

# 1. Precision (정밀도)
# 정밀도는 모델이 추출한 키워드 중에서 실제 키워드와 일치하는 키워드의 비율을 나타냅니다. 즉, 모델이 예측한 키워드 중에서 얼마나 많은 키워드가 실제로 중요한 키워드인지 측정합니다.
# 2. Recall (재현율)
# 재현율은 실제 키워드 중에서 모델이 얼마나 많은 키워드를 올바르게 예측했는지 측정합니다. 즉, 실제 중요한 키워드 중에서 얼마나 많은 키워드를 모델이 잡아냈는지를 나타냅니다.
# 3. F1-Score (F1 점수)
# F1-Score는 정밀도와 재현율의 조화 평균으로, 두 지표의 균형을 측정합니다. 정밀도와 재현율 사이의 트레이드오프를 균형 있게 고려합니다.
# 4. ROUGE-1
# ROUGE-1은 추출된 키워드와 실제 키워드 간의 단어 단위의 겹침을 측정합니다. 이는 단순히 키워드들 사이에 일치하는 단어의 수를 세어 계산됩니다.
# 5. ROUGE-2
# ROUGE-2는 추출된 키워드와 실제 키워드 간의 2-그램(바이그램) 단위의 겹침을 측정합니다. 2-그램은 두 단어로 이루어진 조합입니다.
# 6. ROUGE-L
# ROUGE-L은 추출된 키워드와 실제 키워드 간의 최장 공통 부분 수열(LCS, Longest Common Subsequence)을 기반으로 한 겹침을 측정합니다. 이는 문장의 구조적 유사성을 반영합니다.


#### 2. textrank + term frequency, term postion, word co-occurence
df_filtered_2 = df_filtered.copy()
df_filtered_2.columns

# TF 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 빈도 계산 함수
def calculate_co_occurrence(sentences, window_size=2): # window size 조절
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i+1, min(i+1+window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Textrank 키워드 추출 함수
def textrank_keywords(title, abstract, top_n=5, beta=0.5):
    # 제목과 초록을 합쳐서 문장으로 분할
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # 단어의 TF 값 계산
    tf = calculate_tf(text)
    
    # 공출현 빈도 계산
    co_occurrence = calculate_co_occurrence(sentences)
    
    # 유사도 행렬 초기화
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    # 유사도 행렬 계산
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity = sum(tf[word] for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        similarity += co_occurrence[(word_i, word_j)] / sum(co_occurrence[(word_i, word)] for word in words)
            similarity_matrix[i][j] = similarity
    
    # 가중치 적용
    for word in tf.keys():
        if word in word_tokenize(title.lower()):
            tf[word] = 1
        elif word in word_tokenize(abstract.lower()):
            tf[word] = beta
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    keywords = []
    for score, sentence in ranked_sentences[:top_n]:
        keywords.extend(word_tokenize(sentence.lower()))
    
    return list(set(keywords))

# 추출된 키워드를 데이터 프레임에 추가
df_filtered_2['extracted_keywords'] = df_filtered_2.apply(lambda row: textrank_keywords(row['title'], row['abstract'], top_n=5, beta=0.5), axis=1)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_2[['abstract', 'keywords', 'extracted_keywords']].head())

# Precision, Recall, F1 계산 함수
def calculate_metrics(extracted, actual):
    extracted_set = set(extracted)
    actual_set = set(actual)
    
    true_positive = len(extracted_set & actual_set)
    false_positive = len(extracted_set - actual_set)
    false_negative = len(actual_set - extracted_set)
    
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

df_filtered_2['metrics'] = df_filtered_2.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)
df_filtered_2[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_2['metrics'].tolist(), index=df_filtered_2.index)

# ROUGE 점수 계산 함수
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_2['rouge'] = df_filtered_2.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_2['rouge1'] = df_filtered_2['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_2['rouge2'] = df_filtered_2['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_2['rougeL'] = df_filtered_2['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result2 = df_filtered_2[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result2)


#### 3. textrank + TP-CoGlo-TextRank(GLove)
df_filtered_3 = df_filtered.copy()

# GloVe 임베딩 로드 함수
def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# GloVe 임베딩 파일 경로
glove_file_path = r'D:\대학원\논문\textrank\rawdata\glove.6B.100d.txt' 
glove_embeddings = load_glove_embeddings(glove_file_path)

# 단어 벡터 간 코사인 유사도 계산 함수
def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

# 단어 간 유사도 계산 함수 (GloVe 사용)
def word_similarity(word1, word2, embeddings):
    if word1 in embeddings and word2 in embeddings:
        return cosine_similarity(embeddings[word1], embeddings[word2])
    else:
        return 0.0  # 임베딩이 없는 경우 유사도를 0으로 설정

# 유사도 행렬 계산 with noise  # 데이터 프레임 크기가 작은 경우 인위적으로 noise 추가
def compute_similarity_matrix(words, embeddings, epsilon=1e-5):
    size = len(words)
    similarity_matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            if i != j:
                similarity = word_similarity(words[i], words[j], embeddings)
                similarity_matrix[i][j] = similarity if similarity > epsilon else epsilon
    return similarity_matrix

# without noise
# def compute_similarity_matrix(words, embeddings):
#     size = len(words)
#     similarity_matrix = np.zeros((size, size))
    
#     for i in range(size):
#         for j in range(size):
#             if i != j:
#                 similarity_matrix[i][j] = word_similarity(words[i], words[j], embeddings)
#     return similarity_matrix

# TP-CoGlo-TextRank 키워드 추출 함수
def tp_coglo_textrank(text, top_n=5, embeddings=None):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    # 유사도 행렬 계산
    similarity_matrix = compute_similarity_matrix(words, embeddings)
    # 그래프 생성 및 PageRank 계산
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph, tol=1e-5, max_iter=2000, dangling=None)  # max_iter를 더 늘리고 tol을 더 낮춤
    # 각 단어의 점수를 계산하여 정렬
    ranked_words = sorted(((scores[i], word) for i, word in enumerate(words)), reverse=True)
    # 상위 키워드 추출
    keywords = []
    seen_words = set()
    for _, word in ranked_words:
        if word not in seen_words and word.isalnum():  # 단어가 이미 본 것이 아니고, 알파벳/숫자로만 이루어진 경우
            keywords.append(word)
            seen_words.add(word)
        if len(keywords) >= top_n:
            break
    return keywords

# DataFrame에서 GloVe 및 TP-CoGlo-TextRank를 사용하여 키워드 추출
df_filtered_3['extracted_keywords'] = df_filtered_3['abstract'].apply(lambda x: tp_coglo_textrank(x, top_n=5, embeddings=glove_embeddings))

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_3[['abstract', 'keywords', 'extracted_keywords']])

# Precision, Recall, F1 계산 함수
def calculate_metrics(extracted, actual):
    extracted_set = set(extracted)
    actual_set = set(actual)
    
    true_positive = len(extracted_set & actual_set)
    false_positive = len(extracted_set - actual_set)
    false_negative = len(actual_set - extracted_set)
    
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

df_filtered_3['metrics'] = df_filtered_3.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)
df_filtered_3[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_3['metrics'].tolist(), index=df_filtered_3.index)

print(df_filtered_3[['abstract', 'keywords', 'extracted_keywords', 'precision', 'recall', 'f1']])

# ROUGE 값 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_3['rouge'] = df_filtered_3.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_3['rouge1'] = df_filtered_3['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_3['rouge2'] = df_filtered_3['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_3['rougeL'] = df_filtered_3['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result3 = df_filtered_3[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

print(df_result3)


#### 4. textrank + Watts-Strogatz model
## abstract에서 keyword를 Textrank를 사용하여 추출
df_filtered_4 = df_filtered.copy()
df_filtered_4.columns

# Watts-Strogatz 그래프를 생성하는 함수
def construct_ws_graph(words, p=0.1, k=4):
    size = len(words)
    
    # k 값이 단어 수보다 큰 경우 조정
    if k > size:
        k = max(1, size // 2)  # k를 단어 수의 절반 이하로 조정
    
    # Initialize a regular ring lattice
    graph = nx.watts_strogatz_graph(size, k, p)
    
    # Add edges based on co-occurrence within a window size
    for i in range(size):
        for j in range(i + 1, min(i + k, size)):
            if words[i] != words[j]:
                graph.add_edge(i, j)
    
    return graph

# 텍스트에서 중요 단어를 추출하는 함수
def calculate_ws_weight(text, top_n=5):
    words = word_tokenize(text.lower())
    graph = construct_ws_graph(words, p=0.1, k=4)
    ws_scores = nx.pagerank(graph)
    
    # words의 인덱스가 아닌, 실제 단어로 키워드를 매핑
    ranked_words = sorted(((ws_scores[i], word) for i, word in enumerate(words) if i in ws_scores), reverse=True)
    keywords = [word for _, word in ranked_words[:top_n]]
    return keywords

# 데이터 프레임에 추출된 키워드를 추가
df_filtered_4['extracted_keywords'] = df_filtered_4['abstract'].apply(lambda x: calculate_ws_weight(x, top_n=5))

# 추출된 키워드를 출력
print(df_filtered_4[['abstract', 'keywords', 'extracted_keywords']])


# Precision, Recall, F1 계산 함수
def calculate_metrics(extracted, actual):
    extracted_set = set(extracted)
    actual_set = set(actual)
    
    true_positive = len(extracted_set & actual_set)
    false_positive = len(extracted_set - actual_set)
    false_negative = len(actual_set - extracted_set)
    
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

df_filtered_4['metrics'] = df_filtered_4.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)
df_filtered_4[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_4['metrics'].tolist(), index=df_filtered_4.index)

print(df_filtered_4[['abstract', 'keyword', 'extracted_keywords', 'precision', 'recall', 'f1']].head())

# ROUGE 값 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_4['rouge'] = df_filtered_4.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_4['rouge1'] = df_filtered_4['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_4['rouge2'] = df_filtered_4['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_4['rougeL'] = df_filtered_4['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result4 = df_filtered_4[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

print(df_result4)


#### 5. textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting 
df_filtered_5 = df_filtered.copy()

#### Double Negation, Mitigation, and Hedges Weighting
def apply_weights(text):
    sentences = sent_tokenize(text)
    weighted_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody']]  # 부정어 목록
        if len(negation_indices) > 1:
            # 두 부정어 사이의 거리
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        for word in words:
            if word in ['sort of', 'kind of', 'a little', 'rather']:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        for word in words:
            if word in ['maybe', 'possibly', 'could', 'might']:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# TF 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 빈도 계산 함수
def calculate_co_occurrence(sentences, window_size=2): # window size 조절
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i+1, min(i+1+window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Textrank 키워드 추출 함수
def textrank_keywords(title, abstract, top_n=5, beta=0.5):
    # 제목과 초록을 합쳐서 문장으로 분할
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(text)
    
    sentences = [s for s, w in weighted_sentences]
    weights = [w for s, w in weighted_sentences]
    
    words = word_tokenize(text.lower())
    
    # 단어의 TF 값 계산
    tf = calculate_tf(text)
    
    # 공출현 빈도 계산
    co_occurrence = calculate_co_occurrence(sentences)
    
    # 유사도 행렬 초기화
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    # 유사도 행렬 계산
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity = sum(tf[word] for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        similarity += co_occurrence[(word_i, word_j)] / sum(co_occurrence[(word_i, word)] for word in words)
            similarity_matrix[i][j] = similarity * ((weights[i] + weights[j]) / 2)  # 가중치 적용
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    keywords = []
    for score, sentence in ranked_sentences[:top_n]:
        keywords.extend(word_tokenize(sentence.lower()))
    
    return list(set(keywords))

# 추출된 키워드를 데이터 프레임에 추가
df_filtered_5['extracted_keywords'] = df_filtered_5.apply(lambda row: textrank_keywords(row['title'], row['abstract'], top_n=5, beta=0.5), axis=1)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_5[['abstract', 'keywords', 'extracted_keywords']])

# Precision, Recall, F1 계산 함수
def calculate_metrics(extracted, actual):
    extracted_set = set(extracted)
    actual_set = set(actual)
    
    true_positive = len(extracted_set & actual_set)
    false_positive = len(extracted_set - actual_set)
    false_negative = len(actual_set - extracted_set)
    
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

df_filtered_5['metrics'] = df_filtered_5.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)
df_filtered_5[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_5['metrics'].tolist(), index=df_filtered_5.index)

# ROUGE 점수 계산 함수
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_5['rouge'] = df_filtered_5.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_5['rouge1'] = df_filtered_5['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_5['rouge2'] = df_filtered_5['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_5['rougeL'] = df_filtered_5['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result5 = df_filtered_5[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result5)


#### 5.1 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Glove
df_filtered_51 = df_filtered.copy()

# GloVe 임베딩 로드 함수
def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# GloVe 임베딩 파일 경로
glove_file_path = r'D:\대학원\논문\textrank\rawdata\glove.6B.100d.txt'
glove_embeddings = load_glove_embeddings(glove_file_path)

# 단어 벡터 간 코사인 유사도 계산 함수
def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

# 단어 간 유사도 계산 함수 (GloVe 사용)
def word_similarity(word1, word2, embeddings):
    if word1 in embeddings and word2 in embeddings:
        return cosine_similarity(embeddings[word1], embeddings[word2])
    else:
        return 0.0  # 임베딩이 없는 경우 유사도를 0으로 설정

# 유사도 행렬 계산 함수
def compute_similarity_matrix(words, embeddings, epsilon=1e-5):
    size = len(words)
    similarity_matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            if i != j:
                similarity = word_similarity(words[i], words[j], embeddings)
                similarity_matrix[i][j] = similarity if similarity > epsilon else epsilon
    return similarity_matrix

# Double Negation, Mitigation, and Hedges Weighting 적용 함수
def apply_weights(text):
    sentences = sent_tokenize(text)
    weighted_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        for word in words:
            if word in ['sort of', 'kind of', 'a little', 'rather']:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        for word in words:
            if word in ['maybe', 'possibly', 'could', 'might']:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# TF 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 빈도 계산 함수
def calculate_co_occurrence(sentences, window_size=2): # window size 조절
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i+1, min(i+1+window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Textrank 키워드 추출 함수 with GloVe
def textrank_keywords_glove(title, abstract, top_n=5, embeddings=None):
    # 제목과 초록을 합쳐서 문장으로 분할
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(text)
    
    sentences = [s for s, w in weighted_sentences]
    weights = [w for s, w in weighted_sentences]
    
    words = word_tokenize(text.lower())
    
    # 단어의 TF 값 계산
    tf = calculate_tf(text)
    
    # 공출현 빈도 계산
    co_occurrence = calculate_co_occurrence(sentences)
    
    # 유사도 행렬 계산
    similarity_matrix = compute_similarity_matrix(words, embeddings)
    
    # 유사도 행렬에 가중치 적용
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity = sum(tf[word] for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        similarity += co_occurrence[(word_i, word_j)] / sum(co_occurrence[(word_i, word)] for word in words)
            similarity_matrix[i][j] *= (weights[i] + weights[j]) / 2  # 가중치 적용
    
    # 그래프 생성 및 PageRank 계산
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    ranked_words = sorted(((scores[i], word) for i, word in enumerate(words)), reverse=True)
    
    # 상위 키워드 추출
    keywords = []
    seen_words = set()
    for _, word in ranked_words:
        if word not in seen_words and word.isalnum():
            keywords.append(word)
            seen_words.add(word)
        if len(keywords) >= top_n:
            break
    
    return keywords

# 추출된 키워드를 데이터 프레임에 추가
df_filtered_51['extracted_keywords'] = df_filtered_51.apply(
lambda row: textrank_keywords_glove(row['title'], row['abstract'], top_n=5, embeddings=glove_embeddings), axis=1)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_51[['abstract', 'keywords', 'extracted_keywords']])

# Precision, Recall, F1 계산 함수
df_filtered_51['metrics'] = df_filtered_51.apply(
    lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

df_filtered_51[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_51['metrics'].tolist(), index=df_filtered_51.index)

# ROUGE 점수 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

df_filtered_51['rouge'] = df_filtered_51.apply(
    lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split(', ')), axis=1
)

df_filtered_51['rouge1'] = df_filtered_51['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_51['rouge2'] = df_filtered_51['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_51['rougeL'] = df_filtered_51['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result51 = df_filtered_51[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result51)


#### 6.textrank + term frequency, term postion, word co-occurence + Infomap (tf, tp 반영해야함)
df_filtered_infomap = df_filtered.copy()

# 공출현 계산 함수
def calculate_co_occurrence(sentences, window_size=2):
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Infomap을 사용한 Textrank 키워드 추출 함수
def infomap_textrank_keywords(title, abstract, top_n=5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
    # 공출현 기반 유사도 행렬 계산
    co_occurrence = calculate_co_occurrence(sentences)
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity = sum(co_occurrence[(word, word)] for word in common_words)
            similarity_matrix[i][j] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap을 사용하여 모듈 찾기
    infomap = Infomap()
    
    for i in range(len(sentences)):
        infomap.add_node(i)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()
    
    # 모듈별로 중요한 단어 선택
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            module_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(module_keywords)
    
    return list(set(keywords))

# DataFrame에 Infomap 기반 Textrank 키워드 추가
df_filtered_infomap['extracted_keywords'] = df_filtered_infomap.apply(lambda row: infomap_textrank_keywords(row['title'], row['abstract'], top_n=5), axis=1)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_infomap[['abstract', 'keywords', 'extracted_keywords']])

# Precision, Recall, F1 계산 함수
df_filtered_infomap['metrics'] = df_filtered_infomap.apply(
    lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

df_filtered_infomap[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_infomap['metrics'].tolist(), index=df_filtered_infomap.index)

# ROUGE 점수 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

df_filtered_infomap['rouge'] = df_filtered_infomap.apply(
    lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split(', ')), axis=1
)

df_filtered_infomap['rouge1'] = df_filtered_infomap['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_infomap['rouge2'] = df_filtered_infomap['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_infomap['rougeL'] = df_filtered_infomap['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result_infomap = df_filtered_infomap[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

print(df_result_infomap)


#### 6.1 textrank + term frequency, term postion, word co-occurence + Infomap + GloVe (tf, tp 반영해야함)
df_filtered_infomap_g = df_filtered.copy()

# GloVe 임베딩 로드 함수
def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# 단어 벡터 간 코사인 유사도 계산 함수
def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

# 단어 간 유사도 계산 함수 (GloVe 사용)
def word_similarity(word1, word2, embeddings):
    if word1 in embeddings and word2 in embeddings:
        return cosine_similarity(embeddings[word1], embeddings[word2])
    else:
        return 0.0  # 임베딩이 없는 경우 유사도를 0으로 설정

# GloVe 임베딩 파일 경로
glove_file_path = r'D:\대학원\논문\textrank\rawdata\glove.6B.100d.txt' # GloVe 파일 경로
glove_embeddings = load_glove_embeddings(glove_file_path)

# Infomap을 사용한 Textrank 키워드 추출 함수
def infomap_textrank_keywords(title, abstract, top_n=5, embeddings=None):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # 유사도 행렬 계산
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i, sentence_i in enumerate(sentences):
        words_i = word_tokenize(sentence_i.lower())
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_j = word_tokenize(sentence_j.lower())
            similarity = sum(word_similarity(word_i, word_j, embeddings) for word_i in words_i for word_j in words_j)
            similarity_matrix[i][j] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap을 사용하여 모듈 찾기
    infomap = Infomap()
    
    for i in range(len(sentences)):
        infomap.add_node(i)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()
    
    # 모듈별로 중요한 단어 선택
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            module_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(module_keywords)
    
    return list(set(keywords))

# DataFrame에 Infomap 기반 Textrank 키워드 추가
df_filtered_infomap_g['extracted_keywords'] = df_filtered_infomap_g.apply(
    lambda row: infomap_textrank_keywords(row['title'], row['abstract'], top_n=5, embeddings=glove_embeddings), axis=1)

# Precision, Recall, F1 계산 함수
df_filtered_infomap_g['metrics'] = df_filtered_infomap_g.apply(
    lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

df_filtered_infomap_g[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_infomap_g['metrics'].tolist(), index=df_filtered_infomap_g.index)

# ROUGE 점수 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

df_filtered_infomap_g['rouge'] = df_filtered_infomap_g.apply(
    lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

df_filtered_infomap_g['rouge1'] = df_filtered_infomap_g['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_infomap_g['rouge2'] = df_filtered_infomap_g['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_infomap_g['rougeL'] = df_filtered_infomap_g['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result_infomap_g = df_filtered_infomap_g[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result_infomap_g)


#### 7. textrank + term frequency, term postion, word co-occurence + Infomap + Hierarchical (tf, tp 반영해야함)
df_filtered_infomap_h = df_filtered.copy()

# 공출현 그래프 생성 함수
def create_co_occurrence_graph(sentences, window_size=2):
    graph = nx.Graph()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                if graph.has_edge(word, words[j]):
                    graph[word][words[j]]['weight'] += 1
                else:
                    graph.add_edge(word, words[j], weight=1)
    return graph

# 확장된 Infomap 기반 키워드 추출 함수
# Infomap 기반 키워드 추출 함수 (계층적 구조 반영)
def hierarchical_infomap_keywords(title, abstract, top_n=5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)

    # 공출현 그래프 생성
    graph = create_co_occurrence_graph(sentences)

    # 노드를 정수로 매핑
    word_to_id = {word: idx for idx, word in enumerate(graph.nodes())}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    # Infomap 알고리즘 적용 (계층적 구조 반영)
    infomap = Infomap()

    for edge in graph.edges(data=True):
        node1 = word_to_id[edge[0]]
        node2 = word_to_id[edge[1]]
        weight = edge[2]['weight']
        infomap.addLink(node1, node2, weight)

    infomap.run()

    # 각 노드가 속한 모듈 정보를 트리 구조에서 추출
    module_assignments = {}
    for node in infomap.tree:
        if node.isLeaf:
            node_name = id_to_word[node.node_id]  # 여기서 node_id로 수정
            module_assignments[node_name] = node.moduleIndex()

    # 모듈별로 단어 점수를 계산
    module_scores = {}
    for node, module_id in module_assignments.items():
        node_score = sum([graph[node][nbr]['weight'] for nbr in graph.neighbors(node)])
        if module_id in module_scores:
            module_scores[module_id].append((node, node_score))
        else:
            module_scores[module_id] = [(node, node_score)]

    # 각 모듈에서 상위 top_n 노드를 추출
    hierarchical_keywords = []
    for module_id, nodes in module_scores.items():
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        keywords = [node for node, score in sorted_nodes[:top_n]]
        hierarchical_keywords.extend(keywords)

    return list(set(hierarchical_keywords))

# 추출된 키워드를 데이터 프레임에 추가
df_filtered_infomap_h['extracted_keywords'] = df_filtered_infomap_h.apply(lambda row: hierarchical_infomap_keywords(row['title'], row['abstract'], top_n=5), axis=1)

# Precision, Recall, F1 계산 함수
def calculate_metrics(extracted, actual):
    extracted_set = set(extracted)
    actual_set = set(actual)
    
    true_positive = len(extracted_set & actual_set)
    false_positive = len(extracted_set - actual_set)
    false_negative = len(actual_set - extracted_set)
    
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

df_filtered_infomap_h['metrics'] = df_filtered_infomap_h.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

df_filtered_infomap_h[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_infomap_h['metrics'].tolist(), index=df_filtered_infomap_h.index)

# ROUGE 점수 계산 함수
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_infomap_h['rouge'] = df_filtered_infomap_h.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_infomap_h['rouge1'] = df_filtered_infomap_h['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_infomap_h['rouge2'] = df_filtered_infomap_h['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_infomap_h['rougeL'] = df_filtered_infomap_h['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result_infomap_h = df_filtered_infomap_h[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result_infomap_h)


#### 8. textrank + term frequency, term postion, word co-occurence + Infomap + Multi Entropy (tf, tp 반영해야함)
df_filtered_infomap_m = df_filtered.copy()

# 공출현 계산 함수
def calculate_co_occurrence(sentences, window_size=2):
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# 엔트로피 계산 함수
def calculate_entropy(prob_dist):
    return -np.sum(prob_dist * np.log(prob_dist + 1e-9))

# 확장된 Infomap 기반 키워드 추출 함수 (확장된 방정식 반영)
def infomap_textrank_keywords_extended(title, abstract, top_n=5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
    # 공출현 기반 유사도 행렬 계산
    co_occurrence = calculate_co_occurrence(sentences)
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity = sum(co_occurrence[(word, word)] for word in common_words)
            similarity_matrix[i][j] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap을 사용하여 모듈 찾기
    infomap = Infomap()
    
    for i in range(len(sentences)):
        infomap.add_node(i)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()
    
    # 각 모듈의 엔트로피를 계층적으로 계산 및 가중치 적용
    module_scores = {}
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            prob_dist = np.array(list(word_freq.values())) / np.sum(list(word_freq.values()))
            entropy = calculate_entropy(prob_dist)
            module_id = node.moduleIndex()

            # 확장 방정식에 따른 가중치 적용
            p_i_star = sum(word_freq.values()) / len(words)
            H_Q = calculate_entropy(prob_dist)
            H_Pi = entropy
            module_weight = p_i_star * (H_Q + H_Pi)

            if module_id in module_scores:
                module_scores[module_id].append((sentence_idx, module_weight))
            else:
                module_scores[module_id] = [(sentence_idx, module_weight)]
    
    # 각 모듈에서 상위 top_n 노드를 추출
    hierarchical_keywords = []
    for module_id, nodes in module_scores.items():
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        keywords = [word_tokenize(sentences[node].lower()) for node, _ in sorted_nodes[:top_n]]
        hierarchical_keywords.extend([word for words in keywords for word in words])
    
    return list(set(hierarchical_keywords))

# DataFrame에 확장된 Infomap 기반 Textrank 키워드 추가
df_filtered_infomap_m['extracted_keywords'] = df_filtered_infomap_m.apply(lambda row: infomap_textrank_keywords_extended(row['title'], row['abstract'], top_n=5), axis=1)

# Precision, Recall, F1 계산 함수
df_filtered_infomap_m['metrics'] = df_filtered_infomap_m.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

df_filtered_infomap_m[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_infomap_m['metrics'].tolist(), index=df_filtered_infomap_m.index)


# Precision, Recall, F1 계산 함수
df_filtered_infomap_m['metrics'] = df_filtered_infomap_m.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split(', ')), axis=1)

df_filtered_infomap_m[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_infomap_m['metrics'].tolist(), index=df_filtered_infomap.index)

# ROUGE 점수 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

df_filtered_infomap_m['rouge'] = df_filtered_infomap_m.apply(
    lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split(', ')), axis=1
)

df_filtered_infomap_m['rouge1'] = df_filtered_infomap_m['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_infomap_m['rouge2'] = df_filtered_infomap_m['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_infomap_m['rougeL'] = df_filtered_infomap_m['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result_infomap_m = df_filtered_infomap_m[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result_infomap_m)




print(df_result1) #### 1. textrank
print(df_result2) #### 2. textrank + term frequency, term postion, word co-occurence
print(df_result3) #### 3. textrank + TP-CoGlo-TextRank(GLove)
print(df_result4) #### 4. textrank + Watts-Strogatz model
print(df_result5) #### 5. textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting 
print(df_result51) #### 5.1 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Glove
print(df_result_infomap) #### 6.textrank + term frequency, term postion, word co-occurence + Infomap
print(df_result_infomap_g) #### 6.1 textrank + term frequency, term postion, word co-occurence + Infomap + GloVe
print(df_result_infomap_h) #### 7. textrank + term frequency, term postion, word co-occurence + Infomap + Hierarchical
print(df_result_infomap_m) #### 8. textrank + term frequency, term postion, word co-occurence + Infomap + Multi Entropy




# 각 DataFrame의 합계 계산 함수
def calculate_sums(df):
    sums = df.sum()
    return sums

# 각 DataFrame의 합계 계산
sums_result1 = calculate_sums(df_result1)
sums_result2 = calculate_sums(df_result2)
sums_result3 = calculate_sums(df_result3)
sums_result4 = calculate_sums(df_result4)
sums_result5 = calculate_sums(df_result5)
sums_result51 = calculate_sums(df_result51)
sums_result_infomap = calculate_sums(df_result_infomap)
sums_result_infomap_g = calculate_sums(df_result_infomap_g)
sums_result_infomap_h = calculate_sums(df_result_infomap_h)
sums_result_infomap_m = calculate_sums(df_result_infomap_m)

# 합계 결과를 사전으로 변환
sums_dict = {
    "result1": sums_result1,
    "result2": sums_result2,
    "result3": sums_result3,
    "result4": sums_result4,
    "result5": sums_result5,
    "result51": sums_result51,
    "infomap": sums_result_infomap,
    "infomap_g": sums_result_infomap_g,
    "infomap_h": sums_result_infomap_h,
    "infomap_m": sums_result_infomap_m
}

# 사전을 DataFrame으로 변환
summary_df = pd.DataFrame(sums_dict)

# 전치 (Transpose)하여 인덱스가 행으로, 열이 컬럼으로 되게 변환
summary_df = summary_df.T
# 필요시 전처리된 데이터를 CSV 파일로 저장
summary_df.to_csv('D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\summary_df_result.csv', index=False)
