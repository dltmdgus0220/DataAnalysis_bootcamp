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

