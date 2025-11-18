import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# nltk.download("stopwords") # 영어 불용어 리스트
# nltk.download("punkt") # 기본 토크나이저
# nltk.download("punkt_tab") # 최신 NLTK에서 추가로 요구
# nltk.download("averaged_perceptron_tagger_eng") # 품사 태거

stop_words = set(stopwords.words("english"))
# print(list(stop_words)[:30]) 
print(type(stop_words), len(stop_words))

example_sentence = "This is an example showing off stop word filtration."

words = word_tokenize(example_sentence) # 토큰화
print(words)

filtered_sentence = [ word for word in words if word.lower() not in stop_words ]
print('원문:', example_sentence)
print('불용어 제거 후:', filtered_sentence)

tags = pos_tag(filtered_sentence)
print(tags)
# [('example', 'NN'), ('showing', 'VBG'), ('stop', 'JJ'), ('word', 'NN'), ('filtration', 'NN'), ('.', '.')]
# NN:명사단수, VBG:동명사/현재분사, JJ:형용사