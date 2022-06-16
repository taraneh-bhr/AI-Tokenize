#---------------------- libraries -------------------------
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import pandas as pd
import csv
#---------------------- Dataset ---------------------------
df=pd.read_csv('../Soha/src/train.csv')
df_t=pd.read_csv('../Soha/src/inference.csv')
header = ['id','label']
lab = ([])
index = -1
#--------------------- All features of Train ---------------
document = [row['query'] for index, row in df.iterrows()]
vector = CountVectorizer()
X = vector.fit_transform(document)
total_words = len(vector.get_feature_names())

for line in df_t['query']:
    new_word_list = word_tokenize(line)
    index += 1
    
    #print (new_word_list)
    #---------------------- Train -----------------------------

    #---------------------- Amoozesh --------------------------
    Amoozesh = [row['query'] for index, row in df.iterrows() if row['label'] == 1]
    vector_a = CountVectorizer()
    X_a = vector_a.fit_transform(Amoozesh)
    tdm_a = pd.DataFrame(X_a.toarray(), columns=vector_a.get_feature_names())

    word_list_a = vector_a.get_feature_names();    
    count_list_a = X_a.toarray().sum(axis=0) 
    frequently_a = dict(zip(word_list_a,count_list_a))

    prob_a = []
    for word, count in zip(word_list_a, count_list_a):
        prob_a.append(count/len(word_list_a))
    dict(zip(word_list_a, prob_a))

    total_counts_words_a = count_list_a.sum(axis=0)

    probability_a_with_ls = []
    for word in new_word_list:
        if word in frequently_a.keys():
            count = frequently_a[word]
        else:
            count = 0
        probability_a_with_ls.append((count + 1)/(total_counts_words_a + total_words))
    dict(zip(new_word_list,probability_a_with_ls))
    total_a = 1
    for i in probability_a_with_ls:
        total_a *= i
    #print (total_a)
    #---------------------- MizEtelaat ------------------------
    Miz = [row['query'] for index, row in df.iterrows() if row['label'] == 2]
    vector_m = CountVectorizer()
    X_m = vector_m.fit_transform(Miz)
    tdm_m = pd.DataFrame(X_m.toarray(), columns=vector_m.get_feature_names())

    word_list_m = vector_m.get_feature_names();    
    count_list_m = X_m.toarray().sum(axis=0) 
    frequently_m = dict(zip(word_list_m,count_list_m))

    probability_m = []
    for word, count in zip(word_list_m, count_list_m):
        probability_m.append(count/len(word_list_m))
    dict(zip(word_list_m, probability_m))

    total_counts_words_m = count_list_m.sum(axis=0)

    probability_m_with_ls = []
    for word in new_word_list:
        if word in frequently_m.keys():
            count = frequently_m[word]
        else:
            count = 0
        probability_m_with_ls.append((count + 1)/(total_counts_words_m + total_words))
    dict(zip(new_word_list,probability_m_with_ls))
    total_m = 1
    for i in probability_m_with_ls:
        total_m *= i
    #print (total_m)
    #---------------------- Ketabkhoone -----------------------
    Ketabkhoone = [row['query'] for index, row in df.iterrows() if row['label'] == 3]
    vector_k = CountVectorizer()
    X_k = vector_k.fit_transform(Ketabkhoone)
    tdm_k = pd.DataFrame(X_k.toarray(), columns=vector_k.get_feature_names())

    word_list_k = vector_k.get_feature_names();    
    count_list_k = X_k.toarray().sum(axis=0) 
    frequntly_k = dict(zip(word_list_k,count_list_k))

    probability_k = []
    for word, count in zip(word_list_k, count_list_k):
        probability_k.append(count/len(word_list_k))
    dict(zip(word_list_k, probability_k))

    total_counts_words_k = count_list_k.sum(axis=0)

    probability_k_with_ls = []
    for word in new_word_list:
        if word in frequntly_k.keys():
            count = frequntly_k[word]
        else:
            count = 0
        probability_k_with_ls.append((count + 1)/(total_counts_words_k + total_words))
    dict(zip(new_word_list,probability_k_with_ls))
    total_k = 1
    for i in probability_k_with_ls:
        total_k *= i
    #print (total_k)
    #---------------------- Enteghad & Pishnehad --------------
    E_P = [row['query'] for index, row in df.iterrows() if row['label'] == 4]
    vector_e = CountVectorizer()
    X_e = vector_e.fit_transform(E_P)
    tdm_e = pd.DataFrame(X_e.toarray(), columns=vector_e.get_feature_names())

    word_list_e = vector_e.get_feature_names();    
    count_list_e = X_e.toarray().sum(axis=0) 
    frequently_e = dict(zip(word_list_e,count_list_e))

    probability_e = []
    for word, count in zip(word_list_e, count_list_e):
        probability_e.append(count/len(word_list_e))
    dict(zip(word_list_e, probability_e))

    total_counts_words_e = count_list_e.sum(axis=0)

    probability_e_with_ls = []
    for word in new_word_list:
        if word in frequently_e.keys():
            count = frequently_e[word]
        else:
            count = 0
        probability_e_with_ls.append((count + 1)/(total_counts_words_e + total_words))
    dict(zip(new_word_list,probability_e_with_ls))
    total_e = 1
    for i in probability_e_with_ls:
        total_e *= i
    #print (total_e)
    #---------------------- Sayer -----------------------------
    Sayer = [row['query'] for index, row in df.iterrows() if row['label'] == 5]
    vector_s = CountVectorizer()
    X_s = vector_s.fit_transform(Sayer)
    tdm_s = pd.DataFrame(X_s.toarray(), columns=vector_s.get_feature_names())

    word_list_s = vector_s.get_feature_names();    
    count_list_s = X_s.toarray().sum(axis=0) 
    frequntly_s = dict(zip(word_list_s,count_list_s))

    probability_s = []
    for word, count in zip(word_list_s, count_list_s):
        probability_s.append(count/len(word_list_s))
    dict(zip(word_list_s, probability_s))

    total_counts_words_s = count_list_s.sum(axis=0)

    probability_s_with_ls = []
    for word in new_word_list:
        if word in frequntly_s.keys():
            count = frequntly_s[word]
        else:
            count = 0
        probability_s_with_ls.append((count + 1)/(total_counts_words_s + total_words))
    dict(zip(new_word_list,probability_s_with_ls))
    total_s = 1
    for i in probability_s_with_ls:
        total_s *= i
    #print (total_s)



    max_t = max(total_a, total_m, total_k,total_e,total_s)
    if(max_t==total_e):
        max_t = total_e
        lab.append([index,4])
    elif(max_t==total_k):
        max_t = total_k
        lab.append([index,3])
    elif(max_t==total_m):
        max_t= total_m
        lab.append([index,2])
    elif(max_t==total_s):
        max_t = total_s
        lab.append([index,5])
    elif(max_t==total_a):
        max_t = total_a
        lab.append([index,1])
    #print(lab)
    with open('../Soha/src/interface_result.csv', 'w') as res:
        writer = csv.writer(res)
        writer.writerow(header)
        writer.writerows(lab)
