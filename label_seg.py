import pandas as pd
import matplotlib.pyplot as plt
import re
import random
import spacy 
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from collections import Counter
from itertools import islice

nlp = spacy.load("en")

userdir = "/home/emily2h/Summer/cg-abstracts"
filename = "vci_1543_abs_tit_key_apr_1_2019_train.csv"

df = pd.read_csv("{}/{}".format(userdir, filename))#, sep='\t', header=None, names = ["Abstract", "Classification"])


keywords2 = open("keywords2.txt", "r").read().splitlines()
keywords2 = [w.lower() for w in keywords2]

print(df.info())

addtolabelcol = lambda row: 1\
        if '2' in row\
        else 0
df['true_label2'] = df['Classification'].apply(addtolabelcol)


remKeyWords = lambda row: re.sub(r'\|\|.*\|\|', '', row)
df['Abstract'] = df['Abstract'].apply(remKeyWords)


# labeling function 1
def keywords(abstract):
    ab = abstract.lower()
    for w in keywords2:
        if w in abstract: 
            return 1
    return 0

# labeling function 2
def family(abstract):
    ab = abstract.lower()
    result = re.findall(r'(famil|child|relative)', ab)
    if len(result) < 1:
        return 0
    else:
        return 1

# labeling function 3
def seg(abstract):
    ab = abstract.lower()
    result = re.findall(r'(segregat)', ab)
    if len(result) < 1:
        return 0
    else:
        return 1

# labeling function 4
def member(abstract):
    ab = abstract.lower()
    result = re.findall(r'\d{0,5}.{0,30}(member|individual|relativ)', ab)
    if len(result) < 1:
        return 0
    else:
        return 1

# labeling function 5
def carrier(abstract):
    ab = abstract.lower()
    result = re.findall(r'(carri)', ab)
    if len(result) < 1:
        return 0
    else:
        return 1


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

for a in df.Abstract:
    label_i = Counter()
    label_i[keywords(a)] += 1
    label_i[family(a)] += 1
    label_i[seg(a)] += 1
    label_i[member(a)] += 1
    label_i[carrier(a)] += 1
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
true_labels = list(df['true_label2'])

print("predicted labels 0:1", labels.count(0), labels.count(1))
print("true labels 0:1", true_labels.count(0), true_labels.count(1))

df['predicted_label2'] = labels

df_fn = df[(df['true_label2'] == 1) & (df['predicted_label2'] == 0)]
print(df_fn.shape)
df_fn.to_csv(path_or_buf="false_negatives.csv", index=False)
print(df_fn.head())

score()
cm1 = confusion_matrix(df.true_label2, df.predicted_label2)


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

fpr = cm1[1,0]/(cm1[1,0]+cm1[1,1])
print('false positive rate: ', fpr)

average_precision = metrics.average_precision_score(df.true_label2, df.predicted_label2)

precision, recall, thresholds = metrics.precision_recall_curve(df.true_label2, df.predicted_label2)
f1 = metrics.f1_score(df.true_label2, df.predicted_label2)
auc = metrics.auc(recall, precision)

auroc = metrics.roc_auc_score(df.true_label2, df.predicted_label2)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
print('F1 score: {0:0.2f}'.format(f1))
print('AUC: {0:0.2f}'.format(auc))
print('AUROC: {0:0.2f}'.format(auroc))
plt.plot([0,1], [0.5,0.5], linestyle='--')
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision-Recall Curve")
#plt.show()
