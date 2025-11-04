import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
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

