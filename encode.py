import numpy as np
import pandas as pd
import torch
import pickle
from sklearn import preprocessing
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


with open("/home/emily2h/Summer/cg-abstracts/data_train.pickle", "rb") as f:
    D = pickle.load(f)

#print(D)

"""
flat_list = []
for sublist in D:
    #print(sublist)
    for item in sublist:
        flat_list.append(item)
        """


#onehot_encoder = preprocessing.OneHotEncoder(sparse=True)
#onehot_encoded = onehot_encoder.fit_transform(D.reshape(-1,1))

#print(onehot_encoded)
#print(onehot_encoded.shape)
#print(type(onehot_encoded))
flat_list = D

print(D[0])

count_vect = CountVectorizer(stop_words=stopwords.words('english'), max_features=1000)
X_train_counts = count_vect.fit_transform(flat_list)
print(count_vect.vocabulary_)
print(X_train_counts.shape)
print(type(X_train_counts.todense()))
tens = torch.from_numpy(X_train_counts.todense())
print(tens)
print(type(tens))

pickling_on = open("x_train.pickle", "wb")
pickle.dump(tens, pickling_on)
