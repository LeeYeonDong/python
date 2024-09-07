# 토픽 축소
reduced_topics = topic_model.reduce_topics(df['제목'].tolist(), nr_topics = 2)

# 각 문서에 대한 주제 확률 분포의 평균을 계산
df['topic_weight'] = [np.mean(prob) if prob.size > 0 else 0 for prob in probabilities]
df = df[['제목','날짜','topic_weight']]

# 모든 토픽의 키워드 확인
all_topics = topic_model.get_topics()

for topic_num, topic_keywords in all_topics.items():
    print(f"토픽 {topic_num}:")
    for keyword, weight in topic_keywords:
        print(f"  {keyword}: {weight}")
    print("\n")

topic_data = []

# 모든 토픽의 키워드를 DataFrame으로 변환
topic_data = []
for topic_num, topic_keywords in all_topics.items():
    for keyword, weight in topic_keywords:
        topic_data.append({"토픽 번호": topic_num, "키워드": keyword, "가중치": weight})

df_topics = pd.DataFrame(topic_data)
df_topics.to_csv('D:/대학원/논문/소논문/부동산_토픽모델링/df_topics_words0709_topics2.csv', index=False, encoding='cp949')


# CSV 파일 불러오기
df_topics = pd.read_csv('D:/대학원/논문/소논문/부동산_토픽모델링/df_topics_words0709_topics2.csv', encoding='cp949')

# Visualization 
text = df['제목'].to_list()
date = df['날짜'].to_list()
len(text)
len(date)


# BERTopic 함수 적용
topics_over_time = topic_model.topics_over_time(text, date, global_tuning=True, evolution_tuning=True)

# frequency 추출
topics_over_time['Words']
topics_over_time['Frequency']

topics_over_time.to_csv('D:/대학원/논문/소논문/부동산_토픽모델링/topics_over_time0709_topics2.csv', index=False, encoding='utf-8')

topics_over_time = pd.read_csv('D:/대학원/논문/소논문/부동산_토픽모델링/topics_over_time0709_topics2.csv', encoding='utf-8')

# Topic의 빈도를 확인
topic_frequencies = topics_over_time['Topic'].value_counts()
print(topic_frequencies)

# Topic이 -1이 아닌 행만 선택
filtered_11_topics = topics_over_time[topics_over_time['Topic'] != -1]


# Visualization 1
vis2_1 = topic_model.visualize_topics_over_time(filtered_11_topics).show()
vis2_1.show()

# Visualization 2 
# 상대적 빈도(relative frequency)로 생성된 토픽에 액세스
topic_model.get_topic_freq().head()

# 두 번째로 빈번하게 생성된 주제, 즉 Topic 0. -1은 outlier
topic_model.get_topic(0)
topic_model.get_topic_info().head(2) # -1 = outlier

topic_nr = topic_model.get_topic_info().iloc[6]["Topic"] # select a frequent topic
topic_model.get_topic(topic_nr)

vis2_2 = topic_model.visualize_topics(top_n_topics = 2)
vis2_2.show()

# Visualization 3
vis2_3 = topic_model.visualize_barchart(top_n_topics = 2)
vis2_3.show()

# Visualization 4
vis2_4 = topic_model.visualize_term_rank()
vis2_4.show()

# Visualization 5
vis2_5 = topic_model.visualize_hierarchy(top_n_topics = 2)
vis2_5.show()

# Visualization 6
vis2_6 = topic_model.visualize_heatmap(top_n_topics = 2)
vis2_6.show()
