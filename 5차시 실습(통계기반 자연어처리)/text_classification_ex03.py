import numpy as np
import pandas as pd
import re
import os
import json

from konlpy.tag import Okt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

okt = Okt()
