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

