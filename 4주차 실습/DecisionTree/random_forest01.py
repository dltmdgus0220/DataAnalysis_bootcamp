from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import RandomForestClassifier, plot_tree
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import permutation_importance # 변수 중요도를 계산하는 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

