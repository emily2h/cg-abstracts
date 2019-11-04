import pandas as pd
import numpy as np
import pickle
import nltk
import matplotlib.pyplot as plt
import sys
import re
import random
import spacy 
from nltk import RegexpParser
from nltk import sent_tokenize, word_tokenize
from nltk import pos_tag
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from collections import Counter
from itertools import islice

NUM_ROWS = 0 
argv = sys.argv[1]

nlp = spacy.load("en")
#userdir = "/home/emily2h/Summer/cg-abstracts"
userdir = "/home/emily2h/Summer/cg-abstracts/data_b"
if argv == 'og_train':
    filename = "vci_1543_abs_tit_key_apr_1_2019_train.csv"
    NUM_ROWS = 1543
    flag = 'og_train'
elif argv == 'train':
    filename = "train_2.tsv"
    NUM_ROWS = 1350 
    flag = 'train'
elif argv == 'test':
    filename = "test_2.tsv"
    NUM_ROWS = 97 
    flag = 'test'
elif argv == 'true_test':
    filename = "true_test2.tsv"
    NUM_ROWS = 347 
    flag = 'true_test'

df = pd.read_csv("{}/{}".format(userdir, filename), sep='\t', header=None, names = ["true_label2", "Abstract"])


df['true_label2'] = df['true_label2'].apply(lambda x : 2 if x == 0 else 1)


keywords2 = open("keywords2.txt", "r").read().splitlines()
keywords2 = [w.lower() for w in keywords2]

print(df.info())

"""
addtolabelcol = lambda row: 1\
        if '2' in row\
        else 0
df['true_label2'] = df['Classification'].apply(addtolabelcol)


remKeyWords = lambda row: re.sub(r'\|\|.*\|\|', '', row)
df['Abstract'] = df['Abstract'].apply(remKeyWords)

"""

# labeling function 1
def keywords(abstract):
    ab = abstract.lower()
    count = 0
    for w in keywords2:
        if w.lower() in ab: 
            count += 1
    if count >= 3:
        return 1
    elif count == 1 or count == 2:
        return 0
    return 2

# labeling function 2
def family(abstract):
    ab = abstract.lower()
    result = re.findall(r'(family|families|child|relative|pedigree)', ab)
    if len(result) >= 2:
        return 1
    elif len(result) == 1:
        return 0
    else:
        return 2

# labeling function 3
def seg(abstract):
    ab = abstract.lower()

    truelab = lambda row : row.true_label2 if row.Abstract == abstract else None
    p = df.apply(truelab, axis=1)

    result = re.findall(r'(segregat)', ab)
    #if len(result) >= 2:
    if len(result) >= 1:
        print('\n')
        print(result)
        print(p.dropna())
        print('\n')
        return 1
    #elif len(result) == 1:
    #    return 0
    else:
        return 2

# labeling function 4
def member(abstract):
    ab = abstract.lower()
 #   truelab = lambda row : row.true_label2 if row.Abstract == abstract else None
#    p = df.apply(truelab, axis=1)

    result = re.findall(r'\d{0,5}.{0,30}(member|individual|relativ|kindred|famil)', ab)
    #print(result)
    #print(p.dropna())

    if len(result) >= 2:
        return 1
    elif len(result) == 1:
        return 0
    else:
        return 2

def unrelated(abstract):
    ab = abstract.lower()
    result = re.findall(r'unrelate', ab)
    if len(result) == 1:
        return 2
    else: 
        return 0

def nounphrase(abstract):
    ab = abstract.lower()
    ab_words = [word_tokenize(s) for s in sent_tokenize(ab)]
    pos_tagged = []
    for sent in ab_words:
        pos_tagged.append(pos_tag(sent))
    chunk_grammar = "NP: {<CD><JJ.?>*<NN.?>}"
    chunk_parser = RegexpParser(chunk_grammar)
    np_chunked = []
    for sentence in pos_tagged:
        np_chunked.append(chunk_parser.parse(sentence))
    most_com = np_chunk_counter(np_chunked)
    count = 0
    for chunk in most_com:
        for word in chunk[0]:
            if word[1] == 'NN' and word[0] == 'family' or word[1] == 'NNS' and word[0] == 'families':
                count += 2
            if word[1] == 'JJ' and word[0] == 'unrelated':
                count -= 10
            if word[1] == 'NN' and word[0] == 'proband' or word[1] == 'NNS' and word[0] == 'probands':
                count += 1
            if word[1] == 'NN' and word[0] == 'relative' or word[1] == 'NNS' and word[0] == 'relatives':
                count += 2
            if word[1] == 'NN' and word[0] == 'pedigree' or word[1] == 'NNS' and word[0] == 'pedigrees':
                count += 2
            if word[1] == 'NN' and word[0] == 'kindred' or word[1] == 'NNS' and word[0] == 'kindreds':
                count += 2
                #print("chunk9!",word)
    #print('\n')
    if count >= 2:
        return 1
    elif count == 1:
        return 0
    elif count <= 0:
        return 2


# labeling function 5
def carrier(abstract):
    ab = abstract.lower()
    result = re.findall(r'(carri)', ab)
    if len(result) >= 2:
        return 1
    elif len(result) == 1:
        return 0
    else:
        return 2

def np_chunk_counter(chunked_sentences):

    # create a list to hold chunks
    chunks = list()

    # for-loop through each chunked sentence to extract noun phrase chunks
    for chunked_sentence in chunked_sentences:
        for subtree in chunked_sentence.subtrees(filter=lambda t: t.label() == 'NP'):
            chunks.append(tuple(subtree))

    # create a Counter object
    chunk_counter = Counter()

    # for-loop through the list of chunks
    for chunk in chunks:
        # increase counter of specific chunk by 1
        chunk_counter[chunk] += 1

    # return 30 most frequent chunks
    return chunk_counter.most_common(30)

# returns random label
#def random_label3():
#    return random.randrange(0, 2)

def tiebreaker(majority):
    newmajority = max(majority)  # change from min to max for diff result
    return [newmajority]


def score():
    count = 0
    classify = lambda row: 1\
        if (row.predicted_label2 == 1 and '2' in row.Classification) or (row.predicted_label2 == 0 and '2' not in row.Classification)\
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
lfs = [keywords, family, unrelated, nounphrase, carrier]
data = []

for a in df.Abstract:
    label_i = Counter()
    for lf in lfs:
        data.append(lf(a))
        label_i[lf(a)] += 1
    """
    label_i[keywords(a)] += 1
    label_i[family(a)] += 1
    label_i[seg(a)] += 1
    label_i[member(a)] += 1
    label_i[carrier(a)] += 1
    """
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
data = np.reshape(data, (NUM_ROWS, len(lfs)))
np.set_printoptions(threshold=np.inf)
print(data)

pickling_on = open("label_mat_{}2.pickle".format(flag), "wb")
pickling_on3 = open("labels_{}2.pickle".format(flag), "wb")
pickle.dump(data, pickling_on)
pickle.dump(df['true_label2'].values, pickling_on3)
labels = [x[0][0] for x in tlabels]

#labels = [random.randrange(0, 2) for _ in range(0, len(df['Abstract']))]  #random labels
#print(labels)
true_labels = list(df['true_label2'])

print("predicted labels 2:1", labels.count(2), labels.count(1))
print("true labels 2:1", true_labels.count(2), true_labels.count(1))

df['predicted_label2'] = labels

df_fn = df[(df['true_label2'] == 1) & (df['predicted_label2'] == 0)]
print(df_fn.shape)
df_fn.to_csv(path_or_buf="false_negatives.csv", index=False)
print(df_fn.head())

#score()
cm1 = confusion_matrix(df.true_label2, df.predicted_label2)


total1=sum(sum(cm1))
# from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[1,1]/(cm1[1,1]+cm1[1,0])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Specificity : ', specificity1)

precision1 = cm1[1,1]/(cm1[1,1]+cm1[0,1])
print('precision: ', precision1)

recall1 = cm1[1,1]/(cm1[1,1]+cm1[1,0])
print('recall: ', recall1)

fpr = cm1[0,1]/(cm1[0,1]+cm1[0,0])
print('false positive rate: ', fpr)

average_precision = metrics.average_precision_score(df.true_label2, df.predicted_label2)

precision, recall, thresholds = metrics.precision_recall_curve(df.true_label2, df.predicted_label2)

tru_prec = metrics.precision_score(df.true_label2, df.predicted_label2)
print("truprec:",tru_prec)
f1 = metrics.f1_score(df.true_label2, df.predicted_label2)
auc = metrics.auc(recall, precision)

f = (2 * (precision1 * recall1)/(precision1 + recall1))
auroc = metrics.roc_auc_score(df.true_label2, df.predicted_label2)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
print('F1 score: {0:0.2f}'.format(f1))
print('AUC: {0:0.2f}'.format(auc))
print('AUROC: {0:0.2f}'.format(auroc))
print('F1 score: {0:0.2f}'.format(f))
plt.plot([0,1], [0.5,0.5], linestyle='--')
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision-Recall Curve")
plt.show()

