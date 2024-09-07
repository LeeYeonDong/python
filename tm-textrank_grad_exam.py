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



# Stanza 설정
stanza.download('en')
nlp = stanza.Pipeline('en')

# 파일 경로
file_path = r'D:\대학원\논문\textrank\rawdata\exampledata.xlsx'

# Excel 파일 읽기
df = pd.read_excel(file_path)

# keyword가 존재하는 행 필터링
df_filtered = df.dropna(subset=['keyword'])

# 전처리 결과를 저장할 리스트
processed_abstracts = []

# 각 추상 텍스트에 대해 Stanza를 사용하여 전처리 수행
for text in df_filtered['abstract']:
    doc = nlp(text)
    processed_text = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos in ['VERB', 'NOUN', 'ADJ']:  # 동사, 명사, 형용사만 필터링
                processed_text.append(word.lemma)
    processed_abstracts.append(" ".join(processed_text))

# 전처리된 결과를 데이터프레임의 새 열로 추가
df_filtered['abstract'] = processed_abstracts

# keyword 열에서 가장 많은 키워드 수 계산
max_keywords = df_filtered['keyword'].apply(lambda x: len(x.split(', '))).max()


#### 1. textrank
## abstract에서 keyword를 Textrank를 사용하여 추출
df_filtered_1 = df_filtered.copy()
nltk.download('punkt')

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
print(df_filtered_1[['abstract', 'keyword', 'extracted_keywords']].head())

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

df_filtered_1['metrics'] = df_filtered_1.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)
df_filtered_1[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_1['metrics'].tolist(), index=df_filtered_1.index)

print(df_filtered_1[['abstract', 'keyword', 'extracted_keywords', 'precision', 'recall', 'f1']].head())

# ROUGE 값 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_1['rouge'] = df_filtered_1.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)

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
print(df_filtered_2[['abstract', 'keyword', 'extracted_keywords']].head())

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

df_filtered_2['metrics'] = df_filtered_2.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)
df_filtered_2[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_2['metrics'].tolist(), index=df_filtered_2.index)

# ROUGE 점수 계산 함수
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_2['rouge'] = df_filtered_2.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_2['rouge1'] = df_filtered_2['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_2['rouge2'] = df_filtered_2['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_2['rougeL'] = df_filtered_2['rouge'].apply(lambda x: x['rougeL'].fmeasure)


df_result2 = df_filtered_2[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result2)


#### 3. textrank + TP-CoGlo-TextRank
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

# # without noise
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
    scores = nx.pagerank(nx_graph, tol=1e-6, max_iter=1000, dangling=None)  # max_iter를 더 늘리고 tol을 더 낮춤
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
print(df_filtered_3[['abstract', 'keyword', 'extracted_keywords']].head())

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

df_filtered_3['metrics'] = df_filtered_3.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)
df_filtered_3[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_3['metrics'].tolist(), index=df_filtered_3.index)

print(df_filtered_3[['abstract', 'keyword', 'extracted_keywords', 'precision', 'recall', 'f1']].head())

# ROUGE 값 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_3['rouge'] = df_filtered_3.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_3['rouge1'] = df_filtered_3['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_3['rouge2'] = df_filtered_3['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_3['rougeL'] = df_filtered_3['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result3 = df_filtered_3[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

print(df_result3)


#### 4. textrank + Watts-Strogatz model
## abstract에서 keyword를 Textrank를 사용하여 추출
df_filtered_4 = df_filtered.copy()

## Construct Word Network Graph using Watts-Strogatz model
def construct_ws_graph(words, p=0.1, k=4):
    size = len(words)
    # Initialize a regular ring lattice
    graph = nx.watts_strogatz_graph(size, k, p)
    # Add edges based on co-occurrence within a window size
    for i in range(size):
        for j in range(i + 1, min(i + k, size)):
            if words[i] != words[j]:
                graph.add_edge(i, j)
    return graph

## Calculate Comprehensive Weight
def calculate_ws_weight(text, top_n=5):
    words = word_tokenize(text.lower())
    graph = construct_ws_graph(words, p=0.1, k=4)
    
    ws_scores = nx.pagerank(graph)
    ranked_words = sorted(((ws_scores[i], word) for i, word in enumerate(words)), reverse=True)
    
    keywords = [word for _, word in ranked_words[:top_n]]
    return keywords

# 추출된 키워드를 데이터 프레임에 추가
df_filtered_4['extracted_keywords'] = df_filtered_4['abstract'].apply(lambda x: calculate_ws_weight(x, top_n=5))

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_4[['abstract', 'keyword', 'extracted_keywords']].head())

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

df_filtered_4['metrics'] = df_filtered_4.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)
df_filtered_4[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_4['metrics'].tolist(), index=df_filtered_4.index)

print(df_filtered_4[['abstract', 'keyword', 'extracted_keywords', 'precision', 'recall', 'f1']].head())

# ROUGE 값 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_4['rouge'] = df_filtered_4.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_4['rouge1'] = df_filtered_4['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_4['rouge2'] = df_filtered_4['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_4['rougeL'] = df_filtered_4['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result4 = df_filtered_4[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

print(df_result4)


#### 5. textrank + Double Negation, Mitigation, and Hedges Weighting
df_filtered_5 = df_filtered.copy()

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

# Textrank 키워드 추출 함수
def textrank_keywords(text, top_n=5):
    sentences_with_weights = apply_double_negation_weight(text)
    
    sentences = [s for s, w in sentences_with_weights]
    weights = [w for s, w in sentences_with_weights]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = (X * X.T).toarray()
    
    # 유사도 행렬에 가중치 적용
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            similarity_matrix[i][j] *= (weights[i] + weights[j]) / 2
    
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
df_filtered_5['extracted_keywords'] = df_filtered_5['abstract'].apply(lambda x: textrank_keywords(x, top_n=5))

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_5[['abstract', 'keyword', 'extracted_keywords']].head())

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

df_filtered_5['metrics'] = df_filtered_5.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)
df_filtered_5[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_5['metrics'].tolist(), index=df_filtered_5.index)

print(df_filtered_5[['abstract', 'keyword', 'extracted_keywords', 'precision', 'recall', 'f1']].head())

# ROUGE 값 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_5['rouge'] = df_filtered_5.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_5['rouge1'] = df_filtered_5['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_5['rouge2'] = df_filtered_5['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_5['rougeL'] = df_filtered_5['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result5 = df_filtered_5[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

print(df_result5)


#### 6. textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting
df_filtered_6 = df_filtered.copy()

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
df_filtered_6['extracted_keywords'] = df_filtered_6.apply(lambda row: textrank_keywords(row['title'], row['abstract'], top_n=5, beta=0.5), axis=1)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_6[['abstract', 'keyword', 'extracted_keywords']].head())

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

df_filtered_6['metrics'] = df_filtered_6.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)
df_filtered_6[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_6['metrics'].tolist(), index=df_filtered_6.index)

# ROUGE 점수 계산 함수
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_6['rouge'] = df_filtered_6.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_6['rouge1'] = df_filtered_6['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_6['rouge2'] = df_filtered_6['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_6['rougeL'] = df_filtered_6['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result6 = df_filtered_6[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result6)


#### 7. textrank + term frequency, term postion, word co-occurence + Infomap
df_filtered_infomap = df_filtered.copy()

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
def hierarchical_infomap_keywords(title, abstract, top_n=5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)

    # 공출현 그래프 생성
    graph = create_co_occurrence_graph(sentences)

    # 노드를 정수로 매핑
    word_to_id = {word: idx for idx, word in enumerate(graph.nodes())}

    # Infomap 알고리즘 적용 (계층적 구조 반영)
    infomap = Infomap()

    for edge in graph.edges(data=True):
        node1 = word_to_id[edge[0]]
        node2 = word_to_id[edge[1]]
        weight = edge[2]['weight']
        infomap.addLink(node1, node2, weight)

    infomap.run()

    # 커뮤니티에서 상위 top_n 노드 추출 (계층적 구조 고려)
    communities = infomap.getModules()
    hierarchical_keywords = []

    for module_id, nodes in communities.items():
        node_scores = {node: sum([graph[node][nbr]['weight'] for nbr in graph.neighbors(node)]) for node in nodes}
        sorted_nodes = sorted(node_scores.items(), key=lambda item: item[1], reverse=True)
        keywords = [node for node, score in sorted_nodes[:top_n]]
        hierarchical_keywords.extend(keywords)

    return list(set(hierarchical_keywords))

# 추출된 키워드를 데이터 프레임에 추가
df_filtered_infomap['extracted_keywords'] = df_filtered_infomap.apply(lambda row: hierarchical_infomap_keywords(row['title'], row['abstract'], top_n=5), axis=1)

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

df_filtered_infomap['metrics'] = df_filtered_infomap.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)
df_filtered_infomap[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_infomap['metrics'].tolist(), index=df_filtered_infomap.index)

# ROUGE 점수 계산 함수
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_infomap['rouge'] = df_filtered_infomap.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_infomap['rouge1'] = df_filtered_infomap['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_infomap['rouge2'] = df_filtered_infomap['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_infomap['rougeL'] = df_filtered_infomap['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result_infomap = df_filtered_infomap[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result_infomap)

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

# Infomap 기반 키워드 추출 함수
def infomap_keywords(title, abstract, top_n=5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)

    # 공출현 그래프 생성
    graph = create_co_occurrence_graph(sentences)

    # 노드를 정수로 매핑
    word_to_id = {word: idx for idx, word in enumerate(graph.nodes())}

    # Infomap 알고리즘 적용
    infomap = Infomap()

    for edge in graph.edges(data=True):
        node1 = word_to_id[edge[0]]
        node2 = word_to_id[edge[1]]
        weight = edge[2]['weight']
        infomap.addLink(node1, node2, weight)

    infomap.run()

    # 커뮤니티에서 상위 top_n 노드 추출
    communities = infomap.getModules()
    node_scores = {node: sum([graph[node][nbr]['weight'] for nbr in graph.neighbors(node)]) for node in graph.nodes()}
    sorted_nodes = sorted(node_scores.items(), key=lambda item: item[1], reverse=True)

    # ID를 다시 단어로 변환하여 반환
    keywords = [node for node, score in sorted_nodes[:top_n]]
    return keywords

# 추출된 키워드를 데이터 프레임에 추가
df_filtered_infomap['extracted_keywords'] = df_filtered_infomap.apply(lambda row: infomap_keywords(row['title'], row['abstract'], top_n=5), axis=1)

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

df_filtered_infomap['metrics'] = df_filtered_infomap.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)
df_filtered_infomap[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_infomap['metrics'].tolist(), index=df_filtered_infomap.index)

# ROUGE 점수 계산 함수
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_infomap['rouge'] = df_filtered_infomap.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_infomap['rouge1'] = df_filtered_infomap['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_infomap['rouge2'] = df_filtered_infomap['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_infomap['rougeL'] = df_filtered_infomap['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result_infomap = df_filtered_infomap[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result_infomap)


#### 8. textrank + term frequency, term postion, word co-occurence + Infomap + Hierarchical
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

df_filtered_infomap_h['metrics'] = df_filtered_infomap_h.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)
df_filtered_infomap_h[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_infomap_h['metrics'].tolist(), index=df_filtered_infomap_h.index)

# ROUGE 점수 계산 함수
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_infomap_h['rouge'] = df_filtered_infomap_h.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_infomap_h['rouge1'] = df_filtered_infomap_h['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_infomap_h['rouge2'] = df_filtered_infomap_h['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_infomap_h['rougeL'] = df_filtered_infomap_h['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result_infomap_h = df_filtered_infomap_h[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result_infomap_h)


#### 9. textrank + term frequency, term postion, word co-occurence + Infomap + Hierarchical + Multi Entropy
# 유사도 행렬 생성 함수
df_filtered_infomap_hme = df_filtered.copy()

# TF 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 빈도 계산 함수
def calculate_co_occurrence(sentences, window_size=2):  # window size 조절
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i+1, min(i+1+window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# 유사도 행렬 생성 함수
def create_similarity_matrix(sentences, tf, co_occurrence):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
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
                        similarity += co_occurrence[(word_i, word_j)] / sum(co_occurrence[(word_i, word)] for word in words_i)
            similarity_matrix[i][j] = similarity
    return similarity_matrix

# 엔트로피 계산 함수
def calculate_entropy(probabilities):
    return -sum(p * log2(p) for p in probabilities if p > 0)

# Infomap을 사용한 키워드 추출 및 다중 수준 엔트로피 계산 함수
def textrank_keywords_extended(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    tf = calculate_tf(text)
    co_occurrence = calculate_co_occurrence(sentences)
    similarity_matrix = create_similarity_matrix(sentences, tf, co_occurrence)
    
    infomap = Infomap()

    # 노드와 엣지 추가
    for i in range(len(similarity_matrix)):
        infomap.add_node(i)
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])

    # Infomap 실행
    infomap.run()

    # 첫 번째 수준의 엔트로피 계산
    first_level_modules = infomap.tree.root.children
    q_star = sum(node.flow for node in first_level_modules)
    H_Q = calculate_entropy([node.flow / q_star for node in first_level_modules])

    # 두 번째 수준의 엔트로피 계산
    second_level_entropy = 0
    for node_i in first_level_modules:
        Pi = node_i.flow
        child_modules = node_i.children
        if child_modules:
            p_star_i = sum(child.flow for child in child_modules)
            H_Pi = calculate_entropy([child.flow / p_star_i for child in child_modules])
            second_level_entropy += Pi * (H_Pi + sum(
                child.flow / p_star_i * calculate_entropy([gchild.flow / child.flow for gchild in child.children])
                for child in child_modules
            ))

    # 다중 수준 엔트로피
    multilevel_entropy = q_star * H_Q + second_level_entropy

    # 키워드 추출
    ranked_sentences = sorted(
        ((node.flow, sentences[node.node_id]) for node in infomap.iterLeafNodes()), reverse=True
    )

    keywords = []
    for score, sentence in ranked_sentences[:top_n]:
        keywords.extend(word_tokenize(sentence.lower()))

    return list(set(keywords)), multilevel_entropy

# 추출된 키워드를 데이터 프레임에 추가
df_filtered_infomap_hme['extracted_keywords'] = df_filtered_infomap_hme.apply(
    lambda row: infomap_keywords(row['title'], row['abstract'], top_n=5, beta=0.5), axis=1
)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_infomap_hme[['abstract', 'keyword', 'extracted_keywords']].head())

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

df_filtered_infomap_hme['metrics'] = df_filtered_infomap_hme.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)
df_filtered_infomap_hme[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_infomap_hme['metrics'].tolist(), index=df_filtered_infomap_hme.index)

# ROUGE 점수 계산 함수
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

df_filtered_infomap_hme['rouge'] = df_filtered_infomap_hme.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keyword'].split(', ')), axis=1)

# ROUGE 점수를 DataFrame에 추가
df_filtered_infomap_hme['rouge1'] = df_filtered_infomap_hme['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_infomap_hme['rouge2'] = df_filtered_infomap_hme['rouge'].apply(lambda x: x['rouge2'].fmeasure)
df_filtered_infomap_hme['rougeL'] = df_filtered_infomap_hme['rouge'].apply(lambda x: x['rougeL'].fmeasure)

df_result_infomap_hme = df_filtered_infomap_hme[['precision', 'recall', 'f1', 'rouge1', 'rouge2', 'rougeL']]

# 결과 출력
print(df_result_infomap_hme)


# 결과 출력
print(df_result1)
print(df_result2)
print(df_result3)
print(df_result4)
print(df_result5)
print(df_result6)
print(df_result_infomap)
print(df_result_infomap_h)
print(df_result_infomap_hme)



# 각 DataFrame의 합계 계산 함수 (이전 코드에서 이미 정의됨)
def calculate_sums(df):
    sums = df.sum()
    return sums

# 각 DataFrame의 합계 계산
sums_result1 = calculate_sums(df_result1)
sums_result2 = calculate_sums(df_result2)
sums_result3 = calculate_sums(df_result3)
sums_result4 = calculate_sums(df_result4)
sums_result5 = calculate_sums(df_result5)
sums_result6 = calculate_sums(df_result6)
sums_result_infomap = calculate_sums(df_result_infomap)
sums_result_infomap_h = calculate_sums(df_result_infomap_h)
sums_result_infomap_hme = calculate_sums(df_result_infomap_hme)

# 합계 결과를 사전으로 변환
sums_dict = {
    "result1": sums_result1,
    "result2": sums_result2,
    "result3": sums_result3,
    "result4": sums_result4,
    "result5": sums_result5,
    "result6": sums_result6,
    "infomap": sums_result_infomap,
    "infomap_h": sums_result_infomap_h,
    "infomap_hme": sums_result_infomap_hme,
}

# 사전을 DataFrame으로 변환
summary_df = pd.DataFrame(sums_dict)

# 전치 (Transpose)하여 인덱스가 행으로, 열이 컬럼으로 되게 변환
summary_df = summary_df.T