# library import
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# example
df_d = pd.read_excel(f"D:/대학원/논문/소논문/부동산_감정사전/re_df.xlsx", engine='openpyxl')

df = df_d

df_s = pd.read_csv('D:/대학원/논문/소논문/부동산_감정사전/re_df.csv', encoding='cp949')

df = df_s

df = df.sample(frac=0.1)
df.columns
df.shape
df.head

selected_df = df[df["sen_pol"] != 0]
selected_df.columns

series = pd.Series(selected_df["sen_pol"])
table = series.value_counts()
table

##
word = selected_df["word"]

title = selected_df['제목'].str.replace("[^가-힣\s]", "", regex=True)

##
polarity = selected_df["sen_pol"] # 양수 음수 이진법으로 분류해야함

polarity = selected_df["sen_pol"] # 양수 음수 이진법으로 분류해야함


# text data tokenize and to sequence
tokenizer = Tokenizer()
tokenizer.fit_on_texts(word)
sequences = tokenizer.texts_to_sequences(word)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(title)
sequences = tokenizer.texts_to_sequences(title)


# sequence pading
max_len = max(len(seq) for seq in sequences)
padded_sq = pad_sequences(sequences, maxlen = max_len, padding = "post")


# 조기 종료 설정 (성능 향상이 멈출 경우 훈련 중지)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 데이터 분리 (80% 훈련 데이터, 20% 검증 데이터)
X = np.array(padded_sq)
y = np.array(polarity)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("훈련 데이터 크기:", X_train.shape, y_train.shape)
print("검증 데이터 크기:", X_val.shape, y_val.shape)


# 과적합 방지 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.regularizers import l1, l2

def create_regularized_model(dropout_rate=0.5):
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))),
        Dropout(dropout_rate),  # 동적 드롭아웃 비율
        Bidirectional(LSTM(64)),
        Dropout(dropout_rate),  # 동적 드롭아웃 비율
        Dense(1, activation='sigmoid', kernel_regularizer=l1(0.01))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_regularized_model()
model.summary()

# 모델 래퍼 생성
model = KerasClassifier(build_fn=create_regularized_model, verbose=0)


# 그리드 서치 파라미터 설정
param_grid = {
    'epochs': [5, 10, 15],
    'batch_size': [1, 16, 32, 64],
    'dropout_rate': [0.1, 0.3, 0.5]
}


# 그리드 서치 수행
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2)
grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=2)

# 최적 파라미터 및 결과 출력
print("최적 파라미터: ", grid_result.best_params_)
print("최적 점수: ", grid_result.best_score_)
# 최적 파라미터:  {'batch_size': 64, 'dropout_rate': 0.1, 'epochs': 10} 
# 최적 점수:  0.931475818157196

# GridSearchCV 시각화
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# GridSearchCV 결과를 DataFrame으로 변환
results = pd.DataFrame(grid_result.cv_results_)

# 'dropout_rate'가 0.1인 경우만 필터링
filtered_results = results[results['param_dropout_rate'] == 0.1]

# 필터링된 결과를 사용하여 'batch_size'와 'epochs'에 따른 평균 테스트 점수의 히트맵 생성
pivot_table = filtered_results.pivot(index='param_batch_size', columns='param_epochs', values='mean_test_score')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt=".4f")
plt.title('GridSearchCV Mean Test Scores for dropout_rate=0.1')
plt.xlabel('Epochs')
plt.ylabel('Batch Size')
plt.show()

# 선 그래프를 사용하여 'epochs'에 따른 성능 시각화
# 'epochs' 하이퍼파라미터를 x축, 'mean_test_score'를 y축으로 사용
if 'param_epochs' in results.columns:
    plt.figure(figsize=(10, 6))
    # sns.lineplot 함수를 사용하여 선 그래프 생성
    sns.lineplot(x=results['param_epochs'].astype(int), y='mean_test_score', data=results, marker='o')
    plt.title('Mean Test Scores vs. Epochs (Line Plot)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Test Score')
    plt.grid(True)
    plt.show()

# model training
X = np.array(padded_sq)
y = np.array(polarity)
X.shape
y.shape

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

def create_regularized_model(dropout_rate=0.1): # 'dropout_rate': 0.1
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))),
        Dropout(dropout_rate),  # 동적 드롭아웃 비율
        Bidirectional(LSTM(64)),
        Dropout(dropout_rate),  # 동적 드롭아웃 비율
        Dense(1, activation='sigmoid', kernel_regularizer=l1(0.01))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model_fit = model.fit(X, y, epochs=10, batch_size = 64,  verbose=2, validation_split=0.2, callbacks=[early_stopping])
model_fit.history
# {'loss': [0.18535393476486206, -0.14402799308300018, -0.27826234698295593, -0.48842212557792664, -0.6825524568557739, -0.7462670803070068, -0.938479483127594, -1.2061094045639038, -1.3183915615081787, -1.5613181591033936, -1.7213165760040283, -1.7835204601287842, -2.0598201751708984, -1.63419771194458, -2.386411428451538], 'accuracy': [0.9056046605110168, 0.9767756462097168, 0.9817491769790649, 0.9834651947021484, 0.9847449064254761, 0.9797132015228271, 0.9824762940406799, 0.9849921464920044, 0.9848612546920776, 0.9851230382919312, 0.984890341758728, 0.9847303628921509, 0.98503577709198, 0.9831016063690186, 0.9852393865585327], 'val_loss': [-0.10590380430221558, -0.35286521911621094, -0.45900556445121765, -0.5959840416908264, -1.321149230003357, -1.4023325443267822, -1.8094402551651, -2.371821403503418, -0.30664315819740295, -1.7252914905548096, -3.560300827026367, -3.236971378326416, -3.9054815769195557, -4.244035720825195, -4.777998924255371], 'val_accuracy': [0.9143205881118774, 0.9085039496421814, 0.9084457755088806, 0.9035598039627075, 0.9105398058891296, 0.8946021199226379, 0.9011749625205994, 0.9048394560813904, 0.8997207880020142, 0.9067589640617371, 0.900942325592041, 0.9023964405059814, 0.9053629636764526, 0.9057119488716125, 0.9080967903137207]}

# model training 시각화
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(model_fit.history['loss'])
plt.plot(model_fit.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

#
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# 모델 예측
predictions = model.predict(X)
predictions = np.round(predictions).astype(int)  # 확률을 이진 값으로 변환

# NaN 값이 있는 행의 인덱스 찾기
nan_indexes = np.where(np.isnan(y))[0]

# NaN 값을 포함한 행 제거
X = np.delete(X, nan_indexes, axis=0)
y = np.delete(y, nan_indexes, axis=0)

# 모델 예측 다시 수행
predictions = model.predict(X)
predictions = np.round(predictions).astype(int)  # 확률을 이진 값으로 변환

# 실제 레이블과 예측 레이블 비교
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

#>>> print(f'Precision: {precision}')
# Precision: 0.9912506333060525
#>>> print(f'Recall: {recall}')
# Recall: 0.9818753860407659
#>>> print(f'F1 Score: {f1}')
# F1 Score: 0.9865407365746757

# generate sentiment dict.
sentiment_dict = {}

for word, idx in tokenizer.word_index.items():
    sequence = pad_sequences([[idx]], maxlen = max_len, padding = "post")
    prediction = model_fit.model.predict(sequence)
    sentiment_score = float(prediction[0])
    sentiment_dict[word] = sentiment_score

print(sentiment_dict)

# 딕셔너리를 pandas DataFrame으로 변환
df_sentiment = pd.DataFrame(list(sentiment_dict.items()), columns=['Word', 'Sentiment_Score'])

# DataFrame을 CSV 파일로 저장
import pandas as pd

# 딕셔너리를 pandas DataFrame으로 변환
df_sentiment = pd.DataFrame(list(sentiment_dict.items()), columns=['Word', 'Sentiment_Score'])

# 지정된 경로에 DataFrame을 CSV 파일로 저장
file_path = "D:\\대학원\\논문\\소논문\\부동산_감정사전\\sentiment_scores.csv"
df_sentiment.to_csv(file_path, index=False, encoding='cp949')
