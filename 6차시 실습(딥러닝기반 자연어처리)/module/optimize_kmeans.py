from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def elbow(vector, start:int=1, end:int=10, random_state:int=42):
    inertias = []

    for k in range(start, end+1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init="auto"
        )
        kmeans.fit(vector)
        inertias.append(kmeans.inertia_)
        
    plt.plot(range(start, end+1), inertias, marker="o")
    plt.xlabel("k (클러스터 개수)")
    plt.ylabel("Inertia (SSE)")
    plt.title("Elbow Method")
    plt.show()

    return inertias


def silhouette(vector, start:int=2, end:int=10, random_state:int=42): # X:UMAP 기반 좌표, df
    sil_scores = []

    for k in range(start, end+1):
        kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init="auto"
        )
        labels = kmeans.fit_predict(vector)
        # a(i): 자기 군집 안에서의 평균 거리
        # b(i): 가장 가까운 다른 군집과의 평균 거리
        # (b - a) / max(a, b)
        score = silhouette_score(vector, labels)  # 전체 평균 실루엣 점수
        sil_scores.append(score)
        print(f"k={k}, silhouette_score={score:.4f}")

    plt.plot(range(start, end+1), sil_scores, marker="o")
    plt.xlabel("k (클러스터 개수)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method")
    plt.show()

    return sil_scores


