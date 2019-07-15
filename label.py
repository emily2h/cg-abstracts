import pandas as pd
import random
from sklearn.metrics import confusion_matrix


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

def score():
    count = 0
    classify = lambda row: 1\
        if (row.predicted_label0 == 1 and '0' in row.Classification) or (row.predicted_label0 == 0 and '0' not in row.Classification)\
        else 0
    df['Accuracy'] = df.apply(classify, axis=1)
    print(df['Accuracy'].sum()/len(df['Accuracy']))


labels = []
for a in df.Abstract:
    labels.append(exp_studies_keywords(a))
# labels = [random.randrange(0, 2) for _ in range(0, len(df['Abstract']))] 
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

