import pandas as pd
import re
import random
from sklearn.metrics import confusion_matrix
from collections import Counter


userdir = "/home/emily2h/Summer/cg-abstracts"
filename = "vci_1543_abs_tit_key_apr_1_2019_train.csv"

df = pd.read_csv("{}/{}".format(userdir, filename), sep='\t', header=0)
keywords0 = open("keywords0.txt", "r").read().splitlines()

addtolabelcol = lambda row: 1\
        if '0' in row.Classification\
        else 0
df['true_label0'] = df.apply(addtolabelcol, axis=1)

# labeling function 1
def exp_studies_keywords(abstract):
    for w in keywords0:
        if w in abstract:
            return 1
    return 0

# labeling function 2
def find_percentage(abstract):
    result = re.findall(r'\d{0,2}\-?\d{1,3}\%', abstract)
    if len(result) < 1:
        return 0
    else:
        return 1

"""
# returns random label
def random_label3():
    return random.randrange(0, 2)
"""

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


tlabels = []
for a in df.Abstract:
    label_i = Counter()
    label_i[exp_studies_keywords(a)] += 1
    label_i[find_percentage(a)] += 1
    print(label_i)
    majority = label_i.most_common()
    print(majority)
    if(len(majority) > 1):
        majority = tiebreaker(majority)
    tlabels.append(majority)


print(tlabels)
labels = [x[0][0] for x in tlabels]

# labels = [random.randrange(0, 2) for _ in range(0, len(df['Abstract']))]  # for randomly assigned labels
#print(labels)

print(labels.count(0))
print(labels.count(1))

df['predicted_label0'] = labels


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

