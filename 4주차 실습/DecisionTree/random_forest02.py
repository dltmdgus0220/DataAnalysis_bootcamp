from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance # 변수 중요도를 계산하는 라이브러리
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드
penguins = sns.load_dataset('penguins')

# 2. 데이터 전처리

