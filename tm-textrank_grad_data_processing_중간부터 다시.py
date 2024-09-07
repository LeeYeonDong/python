import pandas as pd
import os
import re
import stanza
import gc

# Stanza 다운로드 및 파이프라인 설정
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=False, use_gpu=False)

# 입력 및 출력 디렉토리 설정
input_dir = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\chunks\\'
output_dir = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\processed_chunks\\'

# 전처리할 파일들의 경로 설정
input_file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14.csv'

# 전처리된 파일들에 대한 경로 설정
output_dir = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\processed_chunks\\'

# 전치사와 관사 리스트 (소문자로 변환하여 저장)
stop_words = {"in", "on", "at", "by", "with", "about", "against", "between", "into", "through", "during", "before", 
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "off", "over", "under", "again", 
              "further", "then", "once", "a", "an", "the"}

# 출력 디렉토리 생성 (존재하지 않을 경우)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 전처리 함수 정의
def preprocess_row(row):
    for column in ['abstract', 'title']:
        text = row[column]
        
        # 텍스트에서 특수 문자 및 괄호 제거
        text_cleaned = re.sub(r'[^A-Za-z0-9\s]', '', str(text))  # 알파벳, 숫자, 공백 외의 문자 제거
        text_cleaned = re.sub(r'[\[\]{}()]', '', text_cleaned)  # 대괄호, 중괄괄호 제거
        text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()  # 다중 공백을 단일 공백으로 변환
        
        # Stanza를 사용한 전처리
        doc = nlp(text_cleaned)
        processed_text = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos in ['VERB', 'NOUN', 'ADJ']:  # 동사, 명사, 형용사만 필터링
                    lemma = word.lemma.lower()
                    # 한 글자 단어, 전치사, 관사 제거
                    if len(lemma) > 1 and lemma not in stop_words:
                        processed_text.append(lemma)
        
        # 전처리된 텍스트를 다시 해당 열에 저장
        row[column] = " ".join(processed_text)
    
    return row

# 전체 파일 목록을 가져오기
all_files = os.listdir(input_dir)
all_files.sort()  # 파일 이름을 기준으로 정렬

# 이미 전처리된 파일 목록 가져오기
processed_files = os.listdir(output_dir)
processed_files_set = set(processed_files)  # 빠른 검색을 위해 set으로 변환

# 전처리가 필요한 파일 목록 필터링
files_to_process = [f for f in all_files if f not in processed_files_set]

# 파일을 청크 단위로 나누어 처리
chunk_size = 100  # 한 번에 처리할 파일 수
num_chunks = len(files_to_process) // chunk_size + 1

for i in range(num_chunks):
    # 현재 청크에 해당하는 파일 목록 선택
    chunk_files = files_to_process[i*chunk_size:(i+1)*chunk_size]

    for filename in chunk_files:
        if filename.endswith('.csv'):
            # 파일 경로 설정
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)
            
            print(f"Processing file: {filename}")
            
            # CSV 파일 불러오기
            df = pd.read_csv(input_file_path)
            
            # 전처리 수행
            df = df.apply(preprocess_row, axis=1)
            
            # 전처리된 데이터를 새로운 CSV 파일로 저장
            df.to_csv(output_file_path, index=False)
            print(f"Processed file saved to: {output_file_path}")
    
    # 메모리 정리
    gc.collect()

print("All files have been processed and saved successfully.")
