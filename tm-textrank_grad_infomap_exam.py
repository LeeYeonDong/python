import nltk
from nltk.tokenize import word_tokenize
import networkx as nx
from infomap import Infomap
from collections import Counter

# NLTK에서 제공하는 데이터 다운로드
nltk.download('punkt')

# 예시 텍스트
text = """
multifaceted human sense touch fundamental direct manipulation technical challenge prevent most teleoperation system provide single modality haptic feedback such force feedback paper postulate ungrounded gripforce fingertipcontactandpressure highfrequency acceleration haptic feedback improve human performance teleoperate pickandplace task subject use teleoperation system consist haptic device wear subject hand remote pr2 humanoid robot motion capture system move object target location subject complete pickandplace task time haptic condition obtain turn gripforce feedback contact feedback acceleration feedback understand object stiffness affect utility feedback half subject complete task flexible plastic cup other use rigid plastic block result indicate addition gripforce feedback gain switch enable subject hold flexible rigid object allow subject manipulate rigid block hold object control motion remote robot hand contact feedback improve ability subject manipulate flexible cup move robot arm space deteriorate ability subject manipulate rigid block contact feedback cause subject hold flexible cup rigid block add acceleration feedback improve subject performance set object hypothesize allow subject feel vibration produce robot motion cause careful complete task study support utility gripforce nd highfrequency acceleration feedback teleoperation system motivate improvement fingertipcontactandpressure feedback
"""

# 공출현 계산 함수
def calculate_co_occurrence(text, window_size=2):
    words = word_tokenize(text.lower())
    co_occurrence = Counter()
    
    for i, word in enumerate(words):
        for j in range(i + 1, min(i + 1 + window_size, len(words))):
            co_occurrence[(word, words[j])] += 1
            co_occurrence[(words[j], word)] += 1
            
    return co_occurrence

# 공출현 계산
co_occurrence = calculate_co_occurrence(text)

# 단어를 정수로 매핑
word_to_id = {word: i for i, word in enumerate(set(word_tokenize(text.lower())))}
id_to_word = {i: word for word, i in word_to_id.items()}

# Infomap 알고리즘 초기화
infomap = Infomap()

# 노드와 엣지를 Infomap 구조에 추가
for word_pair, weight in co_occurrence.items():
    infomap.add_link(word_to_id[word_pair[0]], word_to_id[word_pair[1]], weight)

# Infomap 알고리즘 실행
infomap.run()

# 전체 단어들을 모아서 빈도 계산
all_words = []
for node in infomap.iterTree():
    if node.isLeaf:
        all_words.append(id_to_word[node.physicalId])

# 전체 단어 중 가장 중요한 5개의 단어 확인
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(5)  # 빈도가 높은 상위 5개 단어 선택

# 단어만 추출하여 리스트로 변환
extracted_keywords = [word for word, freq in most_common_words]