# # 데이터 불러오기
# import ijson
# import pandas as pd
# import os

# input_file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14.json'
# output_file_path_csv = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14.csv'

# # CSV로 데이터를 저장하는 함수
# def save_chunk(df, chunk_number):
#     df.to_csv(output_file_path_csv, index=False, mode='a', header=(chunk_number == 0))
#     print(f"Chunk {chunk_number} saved. Number of records in this chunk: {len(df)}")

# # ijson을 사용하여 대용량 JSON 파일 스트리밍 처리
# chunk_size = 2000
# chunk = []
# chunk_number = 0

# with open(input_file_path, 'r', encoding='utf-8') as f:
#     parser = ijson.items(f, 'item')
#     for item in parser:
#         chunk.append(item)
#         if len(chunk) >= chunk_size:
#             print(f"Processing chunk {chunk_number}...")  # 청크 처리 시작 알림
#             df = pd.DataFrame(chunk)
#             save_chunk(df, chunk_number)
#             chunk_number += 1
#             chunk = []
#             print(f"Chunk {chunk_number} processed and saved.")  # 청크 처리 완료 알림

#     # 마지막 남은 청크 처리
#     if chunk:
#         print(f"Processing final chunk {chunk_number}...")  # 마지막 청크 처리 알림
#         df = pd.DataFrame(chunk)
#         save_chunk(df, chunk_number)
#         print(f"Final chunk {chunk_number} processed and saved.")  # 마지막 청크 처리 완료 알림

# print("All chunks processed and saved successfully.")


# # example data 만들기
# import pandas as pd

# # 파일 경로 설정
# input_file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14.csv'
# output_file_path_sample = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_sample.csv'

# # 샘플링할 행의 수
# sample_size = 1000

# # 전체 파일에서 읽을 행 수
# total_rows = 0
# rows_per_chunk = 100000

# print("Calculating total number of rows...")

# # 첫 번째 단계: 전체 파일에서 행 수를 세는 작업 및 샘플링
# sampled_data = pd.DataFrame()  # 빈 데이터프레임 초기화
# header_saved = False  # 헤더가 저장되었는지 확인하는 플래그

# for i, chunk in enumerate(pd.read_csv(input_file_path, chunksize=rows_per_chunk)):
#     total_rows += len(chunk)
    
#     # 현재 청크에서 무작위로 샘플링, 모든 변수를 포함하도록 지정
#     sampled_chunk = chunk.sample(n=min(sample_size, len(chunk)), random_state=42)
#     sampled_data = pd.concat([sampled_data, sampled_chunk], ignore_index=True)
    
#     # 샘플링된 행의 수가 원하는 샘플링 수를 초과하면 중단
#     if len(sampled_data) >= sample_size:
#         sampled_data = sampled_data.head(sample_size)  # 원하는 크기로 자르기
#         break

#     print(f"Processed chunk {i + 1}: {total_rows} rows counted so far.")
#     print(f"Sampled {len(sampled_data)} rows so far.")

# print(f"Total number of rows: {total_rows}")
# print(f"Total sampled rows: {len(sampled_data)}")

# # 세 번째 단계: 샘플링된 데이터를 새로운 CSV 파일로 저장
# print(f"Saving sampled data to {output_file_path_sample}...")

# sampled_data.to_csv(output_file_path_sample, index=False, header=True)

# print(f"Sample of {sample_size} rows saved to {output_file_path_sample}")



import pandas as pd

# 파일 경로 설정
input_file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14.csv'

# 청크 단위로 파일을 불러오기 위한 설정
chunk_size = 100000  # 한 번에 100,000행씩 읽어오기

# 빈 리스트를 만들어 각 청크를 저장
chunks = []

# 청크 단위로 파일을 불러와서 리스트에 저장
for chunk in pd.read_csv(input_file_path, chunksize=chunk_size):
    chunks.append(chunk)

# 모든 청크를 하나의 데이터프레임으로 병합
rawdata = pd.concat(chunks, ignore_index=True)

# 데이터프레임 확인
print(rawdata.info())  # 데이터프레임의 기본 정보 출력
rawdata.columns