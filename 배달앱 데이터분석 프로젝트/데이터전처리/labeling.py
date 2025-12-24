import pandas as pd

# --- 1. 데이터로드 ---

df = pd.read_csv(r'배달앱 데이터분석 프로젝트\데이터\all_reviews_labeled_v6_6.csv', encoding='utf-8-sig')


# --- 2. 별점 기반 라벨링 ---

# 긍정(1) : 별점 4,5점 / 중립(0) : 별점 3점 / 부정(-1) : 별점 1,2점
df.loc[df['score'].isin([4, 5]), 'sentiment_label'] = 1 # 긍정
df.loc[df['score'] == 3, 'sentiment_label'] = 0 # 중립
df.loc[df['score'].isin([1, 2]), 'sentiment_label'] = -1 # 부정

df.loc[df['score'].isin([4, 5]), 'sentiment'] = 'positive' # 긍정
df.loc[df['score'] == 3, 'sentiment'] = 'neutral' # 중립
df.loc[df['score'].isin([1, 2]), 'sentiment'] = 'negative' # 부정


# --- 3. csv 파일로 저장 ---
df.to_csv(r'배달앱 데이터분석 프로젝트\데이터\all_reviews_labeled_score.csv', encoding='utf-8-sig')