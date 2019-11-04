import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import random
import pickle
import numpy as np
import spacy 
import nltk
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
    filename = "train_4.tsv"
    NUM_ROWS = 1350 
    flag = 'train'
elif argv == 'test':
    filename = "test_4.tsv"
    NUM_ROWS = 97 
    flag = 'test'
elif argv == 'true_test':
    filename = "true_test4.tsv"
    NUM_ROWS = 347 
    flag = 'true_test'


df = pd.read_csv("{}/{}".format(userdir, filename), sep='\t', header=None, names = ["true_label4", "Abstract"])

df['true_label4'] = df['true_label4'].apply(lambda x : 2 if x == 0 else 1)


keywords4 = open("keywords4.txt", "r").read().splitlines()
keywords4 = [w.lower() for w in keywords4]

print(df.info())

"""
addtolabelcol = lambda row: 1\
        if '4' in row\
        else 0
df['true_label4'] = df['Classification'].apply(addtolabelcol)


remKeyWords = lambda row: re.sub(r'\|\|.*\|\|', '', row)
df['Abstract'] = df['Abstract'].apply(remKeyWords)
"""


# labeling function 1
def keywords(abstract):
    ab = abstract.lower()
    count = 0

    #truelab = lambda row : row.true_label4 if row.Abstract == abstract else None
    #p = df.apply(truelab, axis=1)
    #print('\n')
    #print(p.dropna())
    for w in keywords4:
        if w.lower() in ab: 
            #print(w)
            count += 1
    if count >= 2:
        return 1
    elif count == 1:
        return 0
    return 2

def number(abstract):
    ab = abstract.lower()
    result = re.findall(r' \d{0,3}\,?\d{1,3}.{0,30}(patient|participant|child|individual|people|proband)', ab)
   # result = re.findall(r'(patient|participant|child|individual|people|proband)', ab)
    #result = re.findall(r' \d{3}',ab)
   # print('\n')
   # print(result)
    if len(result) >= 2:
        return 1
    elif len(result) == 1:
        return 0
    return 2

nouns = ['patient', 'patients', 'participant', 'participants', 'child', 'children', 'individual', 'individuals', 'person', 'people', 'proband', 'probands', 'case', 'cases', 'family', 'families', 'control', 'controls', 'kindred', 'kindreds', 'cohort', 'cohorts', 'group', 'groups', 'referral', 'referrals', 'carrier', 'carriers']
def nounphrase(abstract):
    ab = abstract.lower()
    ab_words = [word_tokenize(s) for s in sent_tokenize(ab)]
    pos_tagged = []
    for sent in ab_words:
        pos_tagged.append(pos_tag(sent))
    chunk_grammar = "NP: {<CD><JJ>*<NN.?>}"
    chunk_parser = RegexpParser(chunk_grammar)
    np_chunked = []
    for sentence in pos_tagged:
        np_chunked.append(chunk_parser.parse(sentence))
    most_com = np_chunk_counter(np_chunked)
    count = 0
    large_num = False 
    #print(most_com)
    for i, chunk in enumerate(most_com):
        for word in chunk[0]:
            if word[1] == 'CD':
                #print('cd:',most_com[i][0][len(chunk[0]) - 1][0])
                if most_com[i][0][len(chunk[0]) - 1][0] in nouns:
                    val = word[0]
                    re.sub(r'\,', '', val)
                #    print('in')
                    try:
                        value = int(val)
                        if value >= 100:
                            count += 2
                            large_num = True
                 #           print("chunk0!",word)
                    except:
                        if 'hundred' in val or 'thousand' in val:
                            count += 2
                            large_num = True
                #            print("chunk0!",word)
            if word[1] == 'JJ' and word[0] == 'unrelated':
                count += 1
            if word[1] == 'NN' and word[0] == 'patient' or word[1] == 'NNS' and word[0] == 'patients':
                count += 1
               # print("chunk1!",word)
            if word[1] == 'NN' and word[0] == 'participant' or word[1] == 'NNS' and word[0] == 'participants':
                count += 1
               # print("chunk2!",word)
            if word[1] == 'NN' and word[0] == 'child' or word[1] == 'NNS' and word[0] == 'children':
                count += 1
              #  print("chunk3!",word)
            if word[1] == 'NN' and word[0] == 'individual' or word[1] == 'NNS' and word[0] == 'individuals':
                count += 1
                #print("chunk4!",word)
            if word[1] == 'NN' and word[0] == 'person' or word[1] == 'NNS' and word[0] == 'people':
                count += 1
               # print("chunk5!",word)
            if word[1] == 'NN' and word[0] == 'proband' or word[1] == 'NNS' and word[0] == 'probands':
                count += 2
              #  print("chunk6!",word)
            if word[1] == 'NN' and word[0] == 'case' or word[1] == 'NNS' and word[0] == 'cases':
                count += 2
             #   print("chunk7!",word)
            if word[1] == 'NN' and word[0] == 'family' or word[1] == 'NNS' and word[0] == 'families':
                count += 1
            #    print("chunk8!",word)
            if word[1] == 'NN' and word[0] == 'control' or word[1] == 'NNS' and word[0] == 'controls':
                count += 2
           #     print("chunk9!",word)
            if word[1] == 'JJ' and word[0] == 'control':
                count += 2
          #      print("control!",word)
            #if word[1] == 'NN' and word[0] == 'mutation' or word[1] == 'NNS' and word[0] == 'mutations':
            #    count += 2
            #    print("chunk10!",word)
            if word[1] == 'NN' and word[0] == 'kindred' or word[1] == 'NNS' and word[0] == 'kindreds':
                count += 1
     #           print("chunk11!",word)
            if word[1] == 'NN' and word[0] == 'cohort' or word[1] == 'NNS' and word[0] == 'cohorts':
                count += 2
      #          print("chunk14!",word)
            if word[1] == 'NN' and word[0] == 'group' or word[1] == 'NNS' and word[0] == 'groups':
                count += 2
       #         print("chunk15!",word)
            if word[1] == 'NN' and word[0] == 'referral' or word[1] == 'NNS' and word[0] == 'referrals':
                count += 2
        #        print("chunk15!",word)
            if word[1] == 'NN' and word[0] == 'carrier' or word[1] == 'NNS' and word[0] == 'carriers':
                count += 2
         #       print("chunk15!",word)
                """
            if word[1] == 'NN' and word[0] == 'variant' or word[1] == 'NNS' and word[0] == 'variants':
                count += 2
                print("chunk12!",word)
            if word[1] == 'NN' and word[0] == 'SNP' or word[1] == 'NNS' and word[0] == 'SNPs':
                count += 2
                print("chunk13!",word)
                """
    #print('\n')
    #print("count",count)
    if count >= 3:
        return 1
    elif count == 1 or count == 2:
        return 0
    elif count <= 0:
        return 2

def nounphrase2(abstract):
    ab = abstract.lower()
    ab_words = [word_tokenize(s) for s in sent_tokenize(ab)]
    pos_tagged = []
    for sent in ab_words:
        pos_tagged.append(pos_tag(sent))
    chunk_grammar = "NP: {<VB.*><DT>?<JJ>*<NN><RB.?>?}"
    #chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"
    #chunk_grammar = "NP: {<DT>?<JJ>*<NN><VB.*><RB.?>?}"
    chunk_parser = RegexpParser(chunk_grammar)
    np_chunked = []
    for sentence in pos_tagged:
        np_chunked.append(chunk_parser.parse(sentence))
    most_com = np_chunk_counter(np_chunked)

    truelab = lambda row : row.true_label4 if row.Abstract == abstract else None
    p = df.apply(truelab, axis=1)
    if p.dropna().values[0] == 1.0:
        print('\n')
        print(p.dropna())
        print(most_com)
        print('\n')
    count = 0
   # print(most_com)
    #print('\n')
   # print("count2",count)
    if count >= 2:
        return 1
    elif count == 1:
        return 0
    elif count <= 0:
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


def number2(abstract):
    ab = abstract.lower()
    result = re.findall(r' \d{0,3}\,?\d{1,3}.{0,30}(case|control)', ab)
   # print(result)
    if len(result) >= 2:
        return 1
    elif len(result) == 1:
        return 0
    return 2


def mutby(abstract):
    ab = abstract.lower()
    #truelab = lambda row : row.true_label3 if row.Abstract == abstract else None
    #newdf = df.apply(truelab, axis=1)
    result = re.findall(r'(mutation).{0,50}(carried by)',ab)
    print(result)
    if len(result) >= 2:
        return 1
    elif len(result) == 1:
        return 0
    return 2


def freqof(abstract):
    ab = abstract.lower()
    #truelab = lambda row : row.true_label4 if row.Abstract == abstract else None
    #p = df.apply(truelab, axis=1)
    #print(abstract)
    result = re.findall(r'(frequency).{0,50}(variant|mutation|SNP|deletion)', ab)
  #  print(result, p.dropna())
    if len(result) >= 1:
        return 1
    else:
        return 2

def casecont(abstract):
    ab = abstract.lower()
    result = re.findall(r'(case).{0,20}(control)', ab)
   # print(result)
    if len(result) >= 1:
        return 1
    return 2

def recurrent(abstract):
    ab = abstract.lower()
    result = re.findall(r'(mutation|systematic).{0,20}(screen)', ab)
   # print(result)
   # print('\n')
    if len(result) >= 1:
        return 1
    return 2

# returns random label
#def random_label3():
#    return random.randrange(0, 2)

def tiebreaker(majority):
    newmajority = max(majority)  # change from min to max for diff result
    return [newmajority]


def score():
    count = 0
    classify = lambda row: 1\
        if (row.predicted_label4 == 1 and '4' in row.Classification) or (row.predicted_label4 == 0 and '4' not in row.Classification)\
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
data = []
lfs = [keywords, number, nounphrase, freqof, casecont, recurrent]

for a in df.Abstract:
    label_i = Counter()
    for lf in lfs:
        data.append(lf(a))
        label_i[lf(a)] += 1
    """
    label_i[keywords(a)] += 1
    label_i[number(a)] += 1
    label_i[number2(a)] += 1
    label_i[mutby(a)] += 1
    label_i[freqof(a)] += 1
    label_i[casecont(a)] += 1
    """
    #label_i[recurrent(a)] += 1
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
print(data[:,5])

pickling_on = open("label_mat_{}4.pickle".format(flag), "wb")
pickling_on3 = open("labels_{}4.pickle".format(flag), "wb")

pickle.dump(data, pickling_on)
pickle.dump(df['true_label4'].values, pickling_on3)

labels = [x[0][0] for x in tlabels]

#labels = [random.randrange(0, 2) for _ in range(0, len(df['Abstract']))]  #random labels
#print(labels)
true_labels = list(df['true_label4'])

print("predicted labels 2:1", labels.count(2), labels.count(1))
print("true labels 2:1", true_labels.count(2), true_labels.count(1))

df['predicted_label4'] = labels

df_fn = df[(df['true_label4'] == 0) & (df['predicted_label4'] == 1)]
print(df_fn.shape)
#df_fn.to_csv(path_or_buf="false_negatives.csv", index=False)
print("fp",df_fn.head())

#score()
cm1 = confusion_matrix(df.true_label4, df.predicted_label4)


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

average_precision = metrics.average_precision_score(df.true_label4, df.predicted_label4)

precision, recall, thresholds = metrics.precision_recall_curve(df.true_label4, df.predicted_label4)

tru_prec = metrics.precision_score(df.true_label4, df.predicted_label4)
print("truprec:",tru_prec)
f1 = metrics.f1_score(df.true_label4, df.predicted_label4)
auc = metrics.auc(recall, precision)

f = (2 * (precision1 * recall1)/(precision1 + recall1))
auroc = metrics.roc_auc_score(df.true_label4, df.predicted_label4)

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

