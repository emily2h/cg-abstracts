import pandas as pd

userdir = "/home/emily2h/Summer/data"
filename = "vci_1543_abs_tit_key_apr_1_2019_train.csv"

df = pd.read_csv("{}/{}".format(userdir, filename), sep = '\t', header=0)

for i in range(5):
    print("i",i)
    #df_i = df["{}".format(i) == df["Classification"]]
    df_i = df[df.Classification.str.contains("{}".format(i))]
    print(df_i.head())
    df_sample = df_i.sample(n=5, random_state=1)
    df_sample.to_csv(path_or_buf="class_{}".format(i), index=False)
