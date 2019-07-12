import pandas as pd


userdir = "/home/emily2h/Summer/data"
filename = "vci_1543_abs_tit_key_apr_1_2019_train.csv"

df = pd.read_csv("{}/{}".format(userdir, filename), sep='\t', header=0)
keywords0 = open("keywords0.txt", "r").read().splitlines()


print(keywords0)
#print(type(df.Abstract[0]))



def exp_studies_keywords(abstract):
    for w in keywords0:
        if w in abstract:
            return 1
    return 0

labels = []
for a in df.Abstract:
    labels.append(exp_studies_keywords(a))

print(labels.count(0))
print(labels.count(1))

df['label0'] = labels
#print(df)


    
