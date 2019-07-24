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
 #   result = re.findall(r'\d{0,2}(\-|\sto\s)?\d{1,3}\s*\%', abstract)
    if len(result) < 1:
        return 0
    else:
        return 1

#labeling function 3
#def 

# returns random label
#def random_label3():
#    return random.randrange(0, 2)
print(re.findall(r'\d*(\-|\sto\s)?\d+\s*\%', "30-50%"))
print(re.findall(r'\d{0,2}\-?\d{1,3}\s*\%', "30-50%"))

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
