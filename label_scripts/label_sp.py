import pandas as pd
import sys
import matplotlib.pyplot as plt
import pickle
import numpy as np
import re
import random
import spacy 
import nltk
from nltk import RegexpParser
from nltk import word_tokenize, sent_tokenize
from nltk import pos_tag
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from collections import Counter
from itertools import islice

#D = pd.read_csv('/home/emily2h/Summer/snorkel/tutorials/cg-abstract2/data/train_copy2pkl.tsv')
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
    filename = "train_3.tsv"
    NUM_ROWS = 1350 
    flag = 'train'
elif argv == 'test':
    filename = "test_3.tsv"
    NUM_ROWS = 97 
    flag = 'test'
elif argv == 'true_test':
    filename = "true_test3.tsv"
    NUM_ROWS = 347 
    flag = 'true_test'


df = pd.read_csv("{}/{}".format(userdir, filename), sep='\t', header=None, names = ["true_label3", "Abstract"])

df['true_label3'] = df['true_label3'].apply(lambda x : 2 if x == 0 else 1)

keywords3 = open("keywords3.txt", "r").read().splitlines()
keywords3 = [w.lower() for w in keywords3]
print(df.info())
"""

addtolabelcol = lambda row: 1\
        if '3' in row\
        else 0
df['true_label3'] = df['Classification'].apply(addtolabelcol)


remKeyWords = lambda row: re.sub(r'\|\|.*\|\|', '', row)
df['Abstract'] = df['Abstract'].apply(remKeyWords)
"""


# labeling function 1
def keywords(abstract):
    ab = abstract.lower()
    count = 0
    keyword = []
    truelab = lambda row : row.true_label3 if row.Abstract == abstract else None
    p = df.apply(truelab, axis=1)
    for w in keywords3:
        if w.lower() in ab:
            keyword.append(w)
            count += 1
    """
    print('\n start')
    print('classification:',p.dropna())
    print('keywords',keyword)
    """

    if count >= 2:
        return 1
    elif count == 1:
        return 0
    return 2
# examination, severity?, overlap, decline?, assay?, test?, genetic testing?, burden?, diagnosis?, variability?
#family history, genotype-phenotype, presented with 
#information, presentation, feature(s), significance, diagnosis, heterogeneity, effectiveness, outcome, expressivity, data, testing, ad?, phenotype, recognition, variability, management, implication, data, evaluation, history
#present with, genotype-phenotype, family history
def nounphrase2(abstract):
    ab = abstract.lower()
    ab_words = [word_tokenize(s) for s in sent_tokenize(ab)]
    pos_tagged = []
    for sent in ab_words:
        pos_tagged.append(pos_tag(sent))
   # chunk_grammar = "NP: {<VB.*><DT>?<JJ>*<NN><RB.?>?}"
    #chunk_grammar = "NP: {<CD|DT>?<JJ.?>*<NN.?><CC><CD|DT>?<JJ.?>*<NN.?>}"
    chunk_grammar = "NP: {<CD|DT>?<JJ.?>*<NN.?>}"
    #chunk_grammar = "NP: {<DT>?<JJ>*<NN><VB.*><RB.?>?}"
    chunk_parser = RegexpParser(chunk_grammar)
    np_chunked = []
    count = 0
    for sentence in pos_tagged:
        np_chunked.append(chunk_parser.parse(sentence))
    #truelab = lambda row : row.true_label3 if row.Abstract == abstract else None
    ab_class = lambda row : row.Abstract if row.Abstract == abstract else None
    #p = df.apply(truelab, axis=1)
    q = df.apply(ab_class, axis=1)

    pd.set_option('display.max_colwidth', -1)
    most_com = np_chunk_counter(np_chunked)
    #print('\n')
    #print(p.dropna())
    #print(q.dropna())
#    print(most_com)
    """
    if p.dropna().values[0] == 1.0:
        print(p.dropna())
        print(most_com)
    """
    words = []
    for i, chunk in enumerate(most_com):
        for word in chunk[0]:
            cur = word[0]
            if word[1] == 'JJ' and (word[0] == 'phenotype' or word[0] == 'genotype-phenotype' or word[0] == 'phenotypic' or word[0] == 'abnormal' or word[0] == 'premature' or word[0] == 'early-onset'):
                count += 1
      #          print(chunk)
                words.append(cur)
                #print("chunk1!",word)
            if word[1] == 'NN' and word[0] == 'phenotype' or word[1] == 'NNS' and word[0] == 'phenotypes' or word[1] == 'NN' and word[0] == 'phenotypic':
                count += 1
                words.append(cur)
                #print("chunk2!",word)
            if word[1] == 'NN' and word[0] == 'correlation' or word[1] == 'NNS' and word[0] == 'correlations':
                count += 1
                words.append(cur)
                #print("chunk2!",chunk)
                """
            if word[1] == 'NN' and word[0] == 'abnormality' or word[1] == 'NNS' and word[0] == 'abnormalities':
                count += 1
                words.append(cur)
                #print("chunk2!",chunk)
                """
            if word[1] == 'NN' and word[0] == 'characterization' or word[1] == 'NNS' and word[0] == 'characterizations' or word[1] == 'NN' and word[0] == 'characterisation' or word[1] == 'NNS' and word[0] == 'characterisations':
                count += 1
                words.append(cur)
                #print("chunk!",chunk)
                """
            if word[1] == 'NN' and word[0] == 'diagnosis' or word[1] == 'NNS' and word[0] == 'diagnoses':
                count += 1
                words.append(cur)
                #print("chunk2!",chunk)
                """
    
    #print("np:",count, words)
#    print('\n')
    if count >= 2:
        return 1
    elif count == 1:
        return 0
    return 2



def case_present(abstract):
    ab = abstract.lower()
    result = re.findall(r'(patient|case).{0,30}(present)', ab)
    result2 = re.findall(r'(present).{0,30}(case)', ab)
    final = result + result2
    #print("case_present:",final)
    if len(final) >= 1:
        return 1
    return 2

combo = ['developed', 'develop', 'treated', 'treatment', 'lesion', 'lesions', 'premature', 'early-onset', 'referred for', 'follow-up', 'abnormality', 'abnormalities', 'abnormal', 'testing'] 
def findinlist(abstract):
    ab = abstract.lower()
    count = 0
    keyword = []
    for w in combo:
        if w in ab:
            keyword.append(w)
            count += 1
    #print('combo of words',keyword)
    if count >= 3:
        return 1
    if count == 1 or count == 2:
        return 0
    return 2


def genphen(abstract):
    ab = abstract.lower()
    result = re.findall(r'(genotype).(phenotype)', ab)
    #print('genotype/phenotype',result)
    if len(result) >= 1:
        return 1
    return 2

# examination, severity?, overlap, decline?, assay?, test?, genetic testing?, burden?, diagnosis?, variability?
def clin(abstract):
    ab = abstract.lower()
    result = re.findall(r'(clinical).{0,25}(examination|severity|overlap|decline|assay|test|genetic test|burden|diagnosis|variability)', ab)
    #print('clinical',result)
    if len(result) >= 1:
        return 1
    return 2

def corr(abstract):
    ab = abstract.lower()
    result = re.findall(r'(relation|correl|associat|confer).{0,25}(genotype)?.{0,15}(phenotype).{0,15}(genotype)?', ab)
    result2 = re.findall(r'(genotype)?.{0,15}(phenotype).{0,15}(genotype)?.{0,25}(relation|correl|associat|confer)', ab)
    final = result + result2
    #print('correlation',final)
    if len(final) >= 1:
        return 1
    return 2

def phen(abstract):
    ab = abstract.lower()
    result = re.findall(r'(phenotype).{0,30}(correl|relati|associat|confer)', ab)
    result2 = re.findall(r'(correl|relati|associat|confer).{0,30}(phenotype)', ab)
    final = result + result2
    #print('phenotype',final)
    if len(final) >= 1:
        return 1
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
        if (row.predicted_label3 == 1 and '3' in row.Classification) or (row.predicted_label3 == 0 and '3' not in row.Classification)\
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
lfs = [keywords, nounphrase2, findinlist, case_present, corr, clin]
data = []

for a in df.Abstract:
    label_i = Counter()
    for lf in lfs:
        value = lf(a)
        data.append(value)
        label_i[value] += 1
        """
    label_i[keywords(a)] += 1
    label_i[case_present(a)] += 1
    label_i[findinlist(a)] += 1
    label_i[genphen(a)] += 1
    label_i[clin(a)] += 1
    """
#    label_i[mutby(a)] += 1
#    label_i[freqof(a)] += 1
#    label_i[casecont(a)] += 1
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
print(data)

pickling_on = open("label_mat_{}3.pickle".format(flag), "wb")
pickling_on3 = open("labels_{}3.pickle".format(flag), "wb")
pickle.dump(data, pickling_on)
pickle.dump(df['true_label3'].values, pickling_on3)

labels = [x[0][0] for x in tlabels]

#labels = [random.randrange(0, 2) for _ in range(0, len(df['Abstract']))]  #random labels
#print(labels)
true_labels = list(df['true_label3'])

print("predicted labels 0:1", labels.count(0), labels.count(1))
print("true labels 0:1", true_labels.count(0), true_labels.count(1))

df['predicted_label3'] = labels

df_fn = df[(df['true_label3'] == 0) & (df['predicted_label3'] == 1)]
print(df_fn.shape)
#df_fn.to_csv(path_or_buf="false_negatives.csv", index=False)
print("fp",df_fn.head())

#score()
cm1 = confusion_matrix(df.true_label3, df.predicted_label3)


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

average_precision = metrics.average_precision_score(df.true_label3, df.predicted_label3)

precision, recall, thresholds = metrics.precision_recall_curve(df.true_label3, df.predicted_label3)

tru_prec = metrics.precision_score(df.true_label3, df.predicted_label3)
print("truprec:",tru_prec)
f1 = metrics.f1_score(df.true_label3, df.predicted_label3)
auc = metrics.auc(recall, precision)

f = (2 * (precision1 * recall1)/(precision1 + recall1))
auroc = metrics.roc_auc_score(df.true_label3, df.predicted_label3)

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

