import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

from matplotlib import rc
import platform

if platform.system() == "Windows":
    plt.rc('font', family="Malgun Gothic")

plt.rcParams['axes.unicode_minus']=False

# 
iris = load_iris() # Bunch객체 리턴
# print(iris)
X = iris.data
Y = iris.target

# X,Y = load_iris(return_X_y=True, as_frame=True)
# print(X)
# print(Y)

print("==붓꽃 데이터셋 정보==")
print(f"데이터 개수: {len(X)}")
print(f"특성(features): {iris.feature_names}")
print(f"클래스(species): {iris.target_names}")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

scaler = StandardScaler().fit(x_tr)
x_tr_scaled = scaler.transform(x_tr)
x_val_scaled = scaler.transform(x_val)

best_cfg = None
best_acc = -1.0

for k in range(3, 31, 2):
    for weights in ['uniform', 'distance']:
        for metric in ['euclidean', 'manhattan']:
            model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
            model.fit(x_tr_scaled, y_tr)
            pred = model.predict(x_val_scaled)
            acc = accuracy_score(y_val, pred)
            # print("K:", k, ", acc:", acc)

            if acc > best_acc:
                best_acc = acc
                best_cfg ={"k":k, "metric":metric, "weights":weights}


print(f"[검증] 최고 정확도 = {best_acc:.3f}, 설정 ={best_cfg}")

scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

knn_model = KNeighborsClassifier(n_neighbors=best_cfg['k'], metric=best_cfg['metric'], weights=best_cfg['weights'])
knn_model.fit(x_train_scaled, y_train)
pred = knn_model.predict(x_test_scaled)
last_acc = accuracy_score(y_test, pred)

print(f"테스트 정확도: {last_acc:.3f}")
print(classification_report(y_test, pred, digits=3))

cm = confusion_matrix(y_test, pred)
dfcm = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot()
sns.heatmap(data=dfcm, annot=True, fmt='d', cmap="Blues", ax=ax)
ax.set_xlabel("예측 값")
ax.set_ylabel("실제 값")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("\n==새로운 데이터 붓꽃 예측===")
new_flower = np.array([[2.1, 1.5, 1.4, 0.2]])
new_flower_scaled = scaler.transform(new_flower)
pred=knn_model.predict(new_flower_scaled)
probabilites = knn_model.predict_proba(new_flower_scaled)
print(pred)
print(probabilites)












