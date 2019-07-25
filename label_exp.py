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


keywords0 = open("keywords0.txt", "r").read().splitlines()

exp = open("experimental_tests.txt", "r").read().splitlines()
exp_test = list(set([w.lower() for w in exp]))
print(exp_test)
print(df.info())

addtolabelcol = lambda row: 1\
        if '0' in row\
        else 0
df['true_label0'] = df['Classification'].apply(addtolabelcol)

lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
lemmas = lemmatizer(u"experimental", u"ADJ")
print(lemmas)

remKeyWords = lambda row: re.sub(r'\|\|.*\|\|', '', row)
df['Abstract'] = df['Abstract'].apply(remKeyWords)


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

# labeling function 4
def in_vitro_before(abstract):
    ab = abstract.lower()
    result = re.findall(r'(in vivo|in vitro|ex vivo).{0,50}(analy|assess|activity)', ab)
    if len(result) < 1:
        return 0
    else:
        #print("HERE IS THE RESULT",result)
        return 1

    
# labeling function 5
def in_vitro_after(abstract):
    ab = abstract.lower()
    result = re.findall(r'(analy|assess).{0,50}(in vivo|in vitro|ex vivo)', ab)
    if len(result) < 1:
        return 0
    else:
        #print("HERE IS THE SECOND RESULT",result)
        return 1

# labeling function 6
def experimental_tests(abstract):
    ab = abstract.lower()
    for test in exp_test:
        if test in ab:
            #print("abstract**",ab)
            #print("this is the test",test)
            return 1
    return 0

# labeling function 7
def elevated(abstract):
    ab = abstract.lower()
    result = re.findall(r'(reduc|increas|elevat|decreas|diminish|high|inflat).{0,50}(compar|activit)', ab)
    if len(result) < 1:
        return 0
    else:
        print(result)
        return 1

# labeling function 8
def elevated_before(abstract):
    ab = abstract.lower()
    result = re.findall(r'(compar|activit).{0,50}(reduc|increas|elevat|decreas|diminish|high|inflat)', ab)
    if len(result) < 1:
        return 0
    else:
        print(result)
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
    label_i[experimental_tests(a)] += 1
    label_i[elevated(a)] += 1
    label_i[elevated_before(a)] += 1
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

fpr = cm1[1,0]/(cm1[1,0]+cm1[1,1])
print('false positive rate: ', fpr)

average_precision = metrics.average_precision_score(df.true_label0, df.predicted_label0)

precision, recall, thresholds = metrics.precision_recall_curve(df.true_label0, df.predicted_label0)
f1 = metrics.f1_score(df.true_label0, df.predicted_label0)
auc = metrics.auc(recall, precision)

auroc = metrics.roc_auc_score(df.true_label0, df.predicted_label0)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
print('F1 score: {0:0.2f}'.format(f1))
print('AUC: {0:0.2f}'.format(auc))
print('AUROC: {0:0.2f}'.format(auroc))
plt.plot([0,1], [0.5,0.5], linestyle='--')
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision-Recall Curve")
plt.show()
