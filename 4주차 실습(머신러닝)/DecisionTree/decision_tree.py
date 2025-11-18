from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)

dt.fit(x_train, y_train)

pred = dt.predict(x_test)
print("Accuracy :", accuracy_score(y_test, pred))
print("\nClassification Report :\n", classification_report(y_test, pred, target_names=iris.target_names))

plt.figure(figsize=(12,6))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.tight_layout()
plt.show()

print(dt.feature_importances_)

for name, imp in sorted(zip(iris.feature_names, dt.feature_importances_), key=lambda x:-x[1]): # 내림차순
    print(f"{name:20s} {imp:.3f}")