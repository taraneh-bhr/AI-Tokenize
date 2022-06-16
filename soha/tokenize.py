from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import re as re
import pandas as pd

df=pd.read_csv("../soha/src/train.csv")
df['query']=df['query'].apply(word_tokenize)

#------------------- train ---------------------

df=pd.read_csv('../soha/src/train.csv')
training_data=df

total = [row['query'] for index,row in training_data.iterrows() if row['label']]

vec_total = CountVectorizer()
X_total = vec_total.fit_transform(total)
tqdm_total = pd.DataFrame(X_total.toarray(), columns=vec_total.get_feature_names())

print(tqdm_total.head())

#darsad faravani har kalame
word_list=vec_total.get_feature_names();
count_list=X_total.toarray().sum(axis=0)
freq_total= dict(zip(word_list,count_list))

print(freq_total)

#nesbat faravani har kalame
prob_total = []
for word, count in zip(word_list,count_list):
    prob_total.append(count/len(word_list))
dict(zip(word_list,prob_total))

#print(prob_total)

#tedad koll data ke dakhel train hast
docs = [row['query'] for index, row in training_data.iterrows()]
vec= CountVectorizer()
X=vec.fit_transform(docs)

#tedad koll hame features
total_features=len(vec.get_feature_names())
print(total_features)
total_cnt= count_list.sum(axis=0)
print(total_cnt)

