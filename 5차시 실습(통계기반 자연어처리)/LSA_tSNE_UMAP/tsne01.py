from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 데이터 로드 (손글씨 숫자 0~9)
digits = load_digits()
X = digits.data      # (n_samples, 64) 8x8 이미지 픽셀
y = digits.target    # 라벨(0~9)

# t-SNE로 2차원 축소
tsne = TSNE(
    n_components=2, perplexity=30,
    learning_rate=200, random_state=42
)

X_2d = tsne.fit_transform(X)

# 시각화
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=10)
plt.title("t-SNE visualization of digits")
plt.show()