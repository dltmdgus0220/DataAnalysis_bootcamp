import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
import platform
if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False


docs = [
"이 브랜드 배송이 너무 빨라서 좋았어요",
"품질은 괜찮은데 배송이 너무 느려요",
"가격이 저렴해서 가성비가 좋아요",
"브랜드 이미지가 세련되고 품질도 좋아요",
"배송도 빠르고 포장도 깔끔해서 만족합니다",
"가격은 비싼 편인데 품질이 좋아요",
]

# tfidf 벡터화
tfidf = TfidfVectorizer(
    max_df=0.8,     # 너무 자주 나오는 단어 무시
    min_df=1,       # 너무 드문 단어 제거 기준
    token_pattern= r"(?u)\b\w+\b"  # 한글 포함 토큰 패턴
)

X = tfidf.fit_transform(docs)
print("TF-IDF shape:", X.shape)  # (문서 수, 단어 수)


# LSA
n_topics = 2
svd = TruncatedSVD(n_components=n_topics, random_state=42)
X_lsa = svd.fit_transform(X)
print("LSA shape:", X_lsa.shape)  # (문서 수, 토픽 수)

lsa_df = pd.DataFrame(X_lsa, columns=[f'topic_{i}' for i in range(n_topics)])
lsa_df['text'] = docs
# print(lsa_df)

topic_name = {
    'topic_0' : '배송',
    'topic_1' : '가격/품질'
}
lsa_df = lsa_df.rename(columns=topic_name)
print(lsa_df)
