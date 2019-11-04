import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
import pickle
import numpy as np
import random
import spacy 
import nltk
from nltk import RegexpParser
from nltk import pos_tag
from nltk import word_tokenize, sent_tokenize
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
    filename = "train_1.tsv"
    NUM_ROWS = 1350 
    flag = 'train'
elif argv == 'test':
    filename = "test_1.tsv"
    NUM_ROWS = 97 
    flag = 'test'
elif argv == 'true_test':
    filename = "true_test1.tsv"
    NUM_ROWS = 347 
    flag = 'true_test'


df = pd.read_csv("{}/{}".format(userdir, filename), sep='\t', header=None, names = ["true_label1", "Abstract"])

print(df.info)
df['true_label1'] = df['true_label1'].apply(lambda x : 2 if x == 0 else 1)
print(df.head)

keywords1 = open("keywords1.txt", "r").read().splitlines()
keywords1 = [w.lower() for w in keywords1]

print(df.info())

"""
addtolabelcol = lambda row: 1\
        if '1' in row\
        else 0
df['true_label1'] = df['Classification'].apply(addtolabelcol)


remKeyWords = lambda row: re.sub(r'\|\|.*\|\|', '', row)
df['Abstract'] = df['Abstract'].apply(remKeyWords)

"""

# labeling function 1
def keywords(abstract):
    ab = abstract.lower()
    count = 0
    for w in keywords1:
        if w.lower() in ab: 
            count += 1
    if count >= 3:
        return 1
    elif count == 1 or count == 2:
        return 0
    return 2

def nounphrase(abstract):
    ab = abstract.lower()
    ab_words = [word_tokenize(s) for s in sent_tokenize(ab)]
    pos_tagged = []
    for sent in ab_words:
        pos_tagged.append(pos_tag(sent))
    chunk_grammar = "NP: {<CD><JJ>?<NN.?>}"
    chunk_parser = RegexpParser(chunk_grammar)
    np_chunked = []
    for sentence in pos_tagged:
        np_chunked.append(chunk_parser.parse(sentence))
    most_com = np_chunk_counter(np_chunked)
    count = 0
    truelab = lambda row : row.true_label1 if row.Abstract == abstract else None
    p = df.apply(truelab, axis=1)
    #print('\n')
    #print(p.dropna())
    #print(most_com)
    words = []
    for chunk in most_com:
        for word in chunk[0]:
            cur = word[0]
            if word[1] == 'NN' and word[0] == 'nonsense' or word[0] == 'missense':
                count += 1
                words.append(cur)
                #print("chunk1!",word)
            if word[1] == 'NN' and word[0] == 'gene' or word[0] == 'allele':
                count += 1
                words.append(cur)
                #print("chunk6!",word)
            if word[1] == 'NNS' and word[0] == 'genes' or word[0] == 'alleles':
                count += 1
                words.append(cur)
                #print("chunk7!",word)
            if word[1] == 'JJ' and word[0] == 'allelic':
                count += 1
                words.append(cur)
                #print("chunk2!",word)
            if word[1] == 'NN' and word[0] == 'proband' or word[1] == 'NNS' and word[0] == 'probands':
                count += 1
                words.append(cur)
                #print("chunk3!",word)
            if word[1] == 'NN' and word[0] == 'mutation' or word[1] == 'NNS' and word[0] == 'mutations':
                count += 1
                words.append(cur)
                #print("chunk4!",word)
            if word[1] == 'NN' and word[0] == 'microdeletion' or word[1] == 'NNS' and word[0] == 'microdeletions':
                count += 1
                words.append(cur)
                #print("chunk5!",word)
            if word[1] == 'NN' and word[0] == 'insertion' or word[1] == 'NNS' and word[0] == 'insertions':
                count += 1
                words.append(cur)
                print("chunk6!",word)
    #print('\n')
    #print('count',count,'  len',len(list(set(words))))
    if len(list(set(words))) >= 2:
        return 1
    elif len(list(set(words))) == 1:
        return 0
    elif len(list(set(words))) <= 0:
        return 2



def allelic(abstract):
    ab = abstract.lower()
    result = re.findall(r'(homozyg|heterozyg).{0,150}(recessive|allelic)', ab)
    result2 = re.findall(r'(recessive|allelic).{0,150}(homozyg|heterozyg)', ab)
    final = result + result2
    if len(final) >= 1:
        return 1
    """
    elif len(final) == 1:
        return 0
        """
    return 2


def het_hom(abstract):
    ab = abstract.lower()
    #truelab = lambda row : row.true_label3 if row.Abstract == abstract else None
    #newdf = df.apply(truelab, axis=1)
    result = re.findall(r'(offspring | cis | trans )', ab)
    if len(result) >= 1:
        #print(result, newdf.dropna())
        return 1
    """
    elif len(result) == 1:
        return 0
    """
    return 2


def prob(abstract):
    ab = abstract.lower()
    #truelab = lambda row : row.true_label3 if row.Abstract == abstract else None
    #p = df.apply(truelab, axis=1)
    #print(abstract)
    result = re.findall(r'(mutation|variant).{0,50}(proband|patient|case)', ab)
    if len(result) >= 1:
        #print(result, newdf.dropna())
        return 1
    """
    elif len(result) == 1:
        return 0
    """
    return 2

def ch(abstract):
    ab = abstract.lower()
    result = re.findall(r'(compound).{0,100}(heterozyg|homozyg)', ab)
    if len(result) >= 1:
        #print(result, newdf.dropna())
        return 1
    """
    elif len(result) == 1:
        return 0
    """
    return 2

# returns random label
#def random_label3():
#    return random.randrange(0, 2)

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



def tiebreaker(majority):
    newmajority = max(majority)  # change from min to max for diff result
    return [newmajority]


def score():
    count = 0
    classify = lambda row: 1\
        if (row.predicted_label1 == 1 and '1' in row.Classification) or (row.predicted_label1 == 0 and '1' not in row.Classification)\
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
lfs = [keywords, nounphrase, allelic, het_hom, ch]
data = []

for a in df.Abstract:
    label_i = Counter()
    for lf in lfs:
        data.append(lf(a))
        label_i[lf(a)] += 1
    """
    label_i[keywords(a)] += 1
    label_i[allelic(a)] += 1
    label_i[het_hom(a)] += 1
    label_i[prob(a)] += 1
    label_i[ch(a)] += 1
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
print("hi*",df['true_label1'].values)
data = np.reshape(data, (NUM_ROWS, len(lfs)))
#np.set_printoptions(threshold=np.inf)
#print(data)
pickling_on = open("label_mat_{}1.pickle".format(flag), "wb")
pickling_on3 = open("labels_{}1.pickle".format(flag), "wb")
pickle.dump(data, pickling_on)
pickle.dump(df['true_label1'].values, pickling_on3)


labels = [x[0][0] for x in tlabels]

#labels = [random.randrange(0, 2) for _ in range(0, len(df['Abstract']))]  #random labels
#print(labels)
true_labels = list(df['true_label1'])

print("predicted labels 0:1", labels.count(0), labels.count(1))
print("true labels 0:1", true_labels.count(0), true_labels.count(1))

df['predicted_label1'] = labels

df_fn = df[(df['true_label1'] == 0) & (df['predicted_label1'] == 1)]
print(df_fn.shape)
#df_fn.to_csv(path_or_buf="false_negatives.csv", index=False)
print("fp",df_fn.head())

#score()
cm1 = confusion_matrix(df.true_label1, df.predicted_label1)


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

average_precision = metrics.average_precision_score(df.true_label1, df.predicted_label1)

precision, recall, thresholds = metrics.precision_recall_curve(df.true_label1, df.predicted_label1)

tru_prec = metrics.precision_score(df.true_label1, df.predicted_label1)
print("truprec:",tru_prec)
f1 = metrics.f1_score(df.true_label1, df.predicted_label1)
auc = metrics.auc(recall, precision)

f = (2 * (precision1 * recall1)/(precision1 + recall1))
auroc = metrics.roc_auc_score(df.true_label1, df.predicted_label1)

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

