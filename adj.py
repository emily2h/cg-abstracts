import spacy
import pandas as pd
from collections import Counter
from collections import OrderedDict 
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

c0, c1, c2, c3, c4 = Counter(), Counter(), Counter(), Counter(), Counter()


def check_adj(word, i):
    c = c0
    if i == 1:
        c = c1
    if i == 2:
        c = c2
    if i == 3:
        c = c3
    if i == 4:
        c = c4

    if word.pos_ == "ADJ":
        c[word.lemma_.lower()] += 1
        #c[word.text.lower()] += 1

def make_cooccur_matrix(df):
    newdf = df.copy()
    names = ['0', '1', '2', '3', '4']
    table = np.zeros((len(names), len(names)), dtype=int)
    count = Counter()

#    newdf['Classification'] = newdf['Classification'].apply(lambda number: number.split())
#    print(newdf.head())
    for line in newdf['Classification']:
        for a in names:
            for b in names:
                if a in line and b in line:
                    table[int(a)][int(b)] += 1
        for word in line:
            count[word] +=1 
    print(count)
    return table
        
    

userdir = "/home/emily2h/Summer/data" # your directory here
filename = "vci_1543_abs_tit_key_apr_1_2019_train.csv"  


df = pd.read_csv("{}/{}".format(userdir, filename), sep = "\t", header=0)
print(df.head())

cooc_mat = make_cooccur_matrix(df)
nm = pd.DataFrame(cooc_mat)
# 1, 3, 4 more commonly cooccur with 0. 
# 3 commonly cooccurs with 1
# 4 commonly cooccurs with 2
# 1 and 4 commonly cooccur with 3
# 2 and 3 commonly cooccur with 4
print(nm)
ax = sn.heatmap(nm, square=True, center=True)
plt.show()


nlp = spacy.load('en')

for i in range(5):
    df_i = df[df.Classification.str.contains("{}".format(i))]
    print(df_i.head())
    pos = df_i['Abstract'].apply(nlp)
    for abstract in pos:
        for w in abstract:
            check_adj(w, i)
print("c0",c0.most_common(10))
print("c1",c1.most_common(10))
print("c2",c2.most_common(10))
print("c3",c3.most_common(10))
print("c4",c4.most_common(10))
    
# make dataframe of adjectives, use counter on classifications of that?

#see if you can do lemmatization
