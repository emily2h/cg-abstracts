import pandas as pd
import re
import random
import spacy 
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from sklearn.metrics import confusion_matrix
from collections import Counter
from itertools import islice

nlp = spacy.load("en")
userdir = "/home/emily2h/Summer/cg-abstracts"
filename = "vci_1543_abs_tit_key_apr_1_2019_train.csv"

df = pd.read_csv("{}/{}".format(userdir, filename), sep='\t', header=0)
keywords0 = open("keywords0.txt", "r").read().splitlines()

addtolabelcol = lambda row: 1\
        if '0' in row.Classification\
        else 0
df['true_label0'] = df.apply(addtolabelcol, axis=1)

lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
lemmas = lemmatizer(u"experimental", u"ADJ")
print(lemmas)

remKeyWords = lambda row: re.sub(r'\|\|.*\|\|', '', row)
df['Abstract'] = df['Abstract'].apply(remKeyWords)

def window(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

# labeling function 1
def exp_studies_keywords(abstract):
    for w in keywords0:
        if w in abstract:
            return 1
    return 0

# labeling function 2
def find_percentage(abstract):
    result = re.findall(r'\d{0,2}\-?\d{1,3}\%', abstract)
 #   result = re.findall(r'\d{0,2}(\-|\sto\s)?\d{1,3}\s*\%', abstract)
    if len(result) < 1:
        return 0
    else:
        return 1

exp_func_keyword = ['experiment', 'experimental', 'function', 'functional']
# labeling function 3
def experimental_functional(abstract):
    ab = nlp(abstract)
    for w in ab:
        if w.lemma_.lower() in exp_func_keyword:
            return 1
    return 0

in_vitro_keyword = [('in', 'vitro'), ('in', 'vivo')]
before = ['analy', 'assess']
after = ['analy', 'activity']
"""
# labeling function 4
def in_vitro_before(abstract):
    ab = [token.text.lower() for token in nlp(abstract)]
    print(ab)
    for pair in window(ab):
        if pair in in_vitro_keyword:
            print(pair)
            return 1
    return 0
"""
def in_vitro_before(abstract):
    result = re.findall(r'(in vivo|in vitro|ex vivo).{0,50}(analy|assess|activity)', abstract)
    if len(result) < 1:
        return 0
    else:
        print("HERE IS THE RESULT",result)
        return 1
in_vitro_before("Conduct an vivo in and in vitro analysis of x.")

    
# labeling function 5
def in_vitro_after(abstract):
    result = re.findall(r'(analy|assess).{0,50}(in vivo|in vitro|ex vivo)', abstract)
    if len(result) < 1:
        return 0
    else:
        print("HERE IS THE SECOND RESULT",result)
        return 1
    

# returns random label
#def random_label3():
#    return random.randrange(0, 2)

def tiebreaker(majority):
    newmajority = min(majority)  # change from min to max for diff result
    return [newmajority]


def score():
    count = 0
    classify = lambda row: 1\
        if (row.predicted_label0 == 1 and '0' in row.Classification) or (row.predicted_label0 == 0 and '0' not in row.Classification)\
        else 0
    df['Accuracy'] = df.apply(classify, axis=1)
    print(df['Accuracy'].sum()/len(df['Accuracy']))

def rem_keywords():
    remKeyWords = lambda row: re.sub(r'\|\|.*\|\|', '', row)
    #remKeyWords = lambda row: re.sub(r'.*\|\|.*\|\|', '', row)    # remove title
    df['Abstract'] = df['Abstract'].apply(remKeyWords)
    print(df.head())

rem_keywords()

tlabels = []

for a in df.Abstract:
    label_i = Counter()
    label_i[exp_studies_keywords(a)] += 1
    label_i[find_percentage(a)] += 1
    label_i[experimental_functional(a)] += 1
    label_i[in_vitro_before(a)] += 1
    label_i[in_vitro_after(a)] += 1
    #majority alg
    majority = label_i.most_common()
    if(len(majority) > 1 and majority[0][1] == majority[1][1]):
        majority = tiebreaker(label_i.most_common())
    """
    if(len(majority) > 1):
        majority = tiebreaker(majority)
    """
    tlabels.append(majority)


#print(tlabels)
labels = [x[0][0] for x in tlabels]

#labels = [random.randrange(0, 2) for _ in range(0, len(df['Abstract']))]  #random labels
#print(labels)
true_labels = list(df['true_label0'])

print("predicted labels 0:1", labels.count(0), labels.count(1))
print("true labels 0:1", true_labels.count(0), true_labels.count(1))

df['predicted_label0'] = labels

df_fn = df[(df['true_label0'] == 1) & (df['predicted_label0'] == 0)]
print(df_fn.shape)
df_fn.to_csv(path_or_buf="false_negatives.csv", index=False)
print(df_fn.head())

score()
cm1 = confusion_matrix(df.true_label0, df.predicted_label0)


total1=sum(sum(cm1))
# from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

precision1 = cm1[0,0]/(cm1[0,0]+cm1[1,1])
print('precision: ', precision1)

recall1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('recall: ', recall1)
