#---------------------- libraries -------------------------
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import pandas as pd
#---------------------- Dataset ---------------------------
df_t=pd.read_csv('../soha/src/test.csv')
sent = input()
new_word_list = word_tokenize(sent)
#---------------------- test -----------------------------

#--------------------- All features of Train ---------------
doc_t = [row['query'] for index, row in df_t.iterrows()]
vec_t = CountVectorizer()
X = vec_t.fit_transform(doc_t)
total_features_t = len(vec_t.get_feature_names())

#---------------------- Amoozesh --------------------------
Amoozesh_t = [row['query'] for index, row in df_t.iterrows() if row['label'] == 1]
vec_a_t = CountVectorizer()
X_a_t = vec_a_t.fit_transform(Amoozesh_t)
tdm_a_t = pd.DataFrame(X_a_t.toarray(), columns=vec_a_t.get_feature_names())

word_list_a_t = vec_a_t.get_feature_names();    
count_list_a_t = X_a_t.toarray().sum(axis=0) 
freq_a_t = dict(zip(word_list_a_t,count_list_a_t))

prob_a_t = []
for word, count in zip(word_list_a_t, count_list_a_t):
    prob_a_t.append(count/len(word_list_a_t))
dict(zip(word_list_a_t, prob_a_t))

total_cnts_features_a_t = count_list_a_t.sum(axis=0)

prob_a_with_ls_t = []
for word in new_word_list:
    if word in freq_a_t.keys():
        count = freq_a_t[word]
    else:
        count = 0
    prob_a_with_ls_t.append((count + 1)/(total_cnts_features_a_t + total_features_t))
dict(zip(new_word_list,prob_a_with_ls_t))
total_a_t = 1
for i in prob_a_with_ls_t:
    total_a_t *= i
print (total_a_t)
#---------------------- MizEtelaat ------------------------
Miz_t = [row['query'] for index, row in df_t.iterrows() if row['label'] == 2]
vec_m_t = CountVectorizer()
X_m_t = vec_m_t.fit_transform(Miz_t)
tdm_m_t = pd.DataFrame(X_m_t.toarray(), columns=vec_m_t.get_feature_names())

word_list_m_t = vec_m_t.get_feature_names();    
count_list_m_t = X_m_t.toarray().sum(axis=0) 
freq_m_t = dict(zip(word_list_m_t,count_list_m_t))

prob_m_t = []
for word, count in zip(word_list_m_t, count_list_m_t):
    prob_m_t.append(count/len(word_list_m_t))
dict(zip(word_list_m_t, prob_m_t))

total_cnts_features_m_t = count_list_m_t.sum(axis=0)

prob_m_with_ls_t = []
for word in new_word_list:
    if word in freq_m_t.keys():
        count = freq_m_t[word]
    else:
        count = 0
    prob_m_with_ls_t.append((count + 1)/(total_cnts_features_m_t + total_features_t))
dict(zip(new_word_list,prob_m_with_ls_t))
total_m_t = 1
for i in prob_m_with_ls_t:
    total_m_t *= i
print (total_m_t)
#---------------------- Ketabkhoone -----------------------
Ketabkhoone_t = [row['query'] for index, row in df_t.iterrows() if row['label'] == 3]
vec_k_t = CountVectorizer()
X_k_t = vec_k_t.fit_transform(Ketabkhoone_t)
tdm_k_t = pd.DataFrame(X_k_t.toarray(), columns=vec_k_t.get_feature_names())

word_list_k_t = vec_k_t.get_feature_names();    
count_list_k_t = X_k_t.toarray().sum(axis=0) 
freq_k_t = dict(zip(word_list_k_t,count_list_k_t))

prob_k_t = []
for word, count in zip(word_list_k_t, count_list_k_t):
    prob_k_t.append(count/len(word_list_k_t))
dict(zip(word_list_k_t, prob_k_t))

total_cnts_features_k_t = count_list_k_t.sum(axis=0)

prob_k_with_ls_t = []
for word in new_word_list:
    if word in freq_k_t.keys():
        count = freq_k_t[word]
    else:
        count = 0
    prob_k_with_ls_t.append((count + 1)/(total_cnts_features_k_t + total_features_t))
dict(zip(new_word_list,prob_k_with_ls_t))
total_k_t = 1
for i in prob_k_with_ls_t:
    total_k_t *= i
print (total_k_t)
#---------------------- Enteghad & Pishnehad --------------
E_P_t = [row['query'] for index, row in df_t.iterrows() if row['label'] == 4]
vec_e_t = CountVectorizer()
X_e_t = vec_e_t.fit_transform(E_P_t)
tdm_e_t = pd.DataFrame(X_e_t.toarray(), columns=vec_e_t.get_feature_names())

word_list_e_t = vec_e_t.get_feature_names();    
count_list_e_t = X_e_t.toarray().sum(axis=0) 
freq_e_t = dict(zip(word_list_e_t,count_list_e_t))

prob_e_t = []
for word, count in zip(word_list_e_t, count_list_e_t):
    prob_e_t.append(count/len(word_list_e_t))
dict(zip(word_list_e_t, prob_e_t))

total_cnts_features_e_t = count_list_e_t.sum(axis=0)

prob_e_with_ls_t = []
for word in new_word_list:
    if word in freq_e_t.keys():
        count = freq_e_t[word]
    else:
        count = 0
    prob_e_with_ls_t.append((count + 1)/(total_cnts_features_e_t + total_features_t))
dict(zip(new_word_list,prob_e_with_ls_t))
total_e_t = 1
for i in prob_e_with_ls_t:
    total_e_t *= i
print (total_e_t)
#---------------------- Sayer -----------------------------
Sayer_t = [row['query'] for index, row in df_t.iterrows() if row['label'] == 5]
vec_s_t = CountVectorizer()
X_s_t = vec_s_t.fit_transform(Sayer_t)
tdm_s_t = pd.DataFrame(X_s_t.toarray(), columns=vec_s_t.get_feature_names())

word_list_s_t = vec_s_t.get_feature_names();    
count_list_s_t = X_s_t.toarray().sum(axis=0) 
freq_s_t = dict(zip(word_list_s_t,count_list_s_t))

prob_s_t = []
for word, count in zip(word_list_s_t, count_list_s_t):
    prob_s_t.append(count/len(word_list_s_t))
dict(zip(word_list_s_t, prob_s_t))

total_cnts_features_s_t = count_list_s_t.sum(axis=0)

prob_s_with_ls_t = []
for word in new_word_list:
    if word in freq_s_t.keys():
        count = freq_s_t[word]
    else:
        count = 0
    prob_s_with_ls_t.append((count + 1)/(total_cnts_features_s_t + total_features_t))
dict(zip(new_word_list,prob_s_with_ls_t))
total_s_t = 1
for i in prob_s_with_ls_t:
    total_s_t *= i
print (total_s_t)
