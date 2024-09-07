import pandas as pd
import os

# 파일 경로 설정
input_dir = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\processed_chunks\\'
output_file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_combined.csv'

# 파일 목록 가져오기
all_files = os.listdir(input_dir)
all_files.sort()  # 파일을 정렬하여 순차적으로 처리

# 빈 데이터프레임 생성
combined_df = pd.DataFrame()

# 파일들을 차례로 읽어와서 통합
for filename in all_files:
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        
        # 읽어온 데이터를 통합 데이터프레임에 추가
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        print(f"Processed {filename}")

# 통합된 데이터프레임을 하나의 CSV 파일로 저장
combined_df.to_csv(output_file_path, index=False)

print(f"All files have been processed and saved to {output_file_path}.")
