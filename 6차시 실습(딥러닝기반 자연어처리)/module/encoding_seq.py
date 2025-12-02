import numpy as np
import pandas as pd
import re
from konlpy.tag import Okt
from collections import Counter
from sklearn.model_selection import train_test_split

# 전역변수
okt = Okt()
counter = Counter()
# 특수 토큰 정의
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

