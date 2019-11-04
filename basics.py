import sys
import metal
import torch
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from scipy import sparse
from sklearn import dummy
from metal.utils import split_data
from metal.end_model import EndModel
from metal.analysis import lf_summary
from metal.label_model import LabelModel
from metal.label_model.baselines import MajorityLabelVoter, MajorityClassVoter
from metal.analysis import confusion_matrix

sys.path.append('../../metal')
flag0, flag = sys.argv[1:3]
print('flag',flag)
print('flag0',flag0)


# put data in data.pickle 

with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/X/x_train.pickle", "rb") as f:
    X = pickle.load(f)

with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/data/data_train.pickle", "rb") as f:
    D = pickle.load(f).tolist()

# put label matrix in label_mat.pickle 
with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/label_mat_train/label_mat_train{}.pickle".format(flag), "rb") as f:
    L = sparse.csr_matrix(pickle.load(f))

# put gold labels in labels.pickle
with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/labels/labels_train{}.pickle".format(flag), "rb") as f:
    Y = pickle.load(f)
"""
with open("data/basics_tutorial.pkl", "rb") as f:
    X, Y, L, D = pickle.load(f)
"""
if flag0 == 'true_test':
    with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/X/x_true_test.pickle".format(flag), "rb") as f:
        X_test = pickle.load(f)

    with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/data/data_true_test.pickle".format(flag), "rb") as f:
        D_test = pickle.load(f)

    with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/label_mat_true_test/label_mat_true_test{}.pickle".format(flag), "rb") as f:
        L_test = pickle.load(f)

    with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/labels/labels_true_test{}.pickle".format(flag), "rb") as f:
        Y_test = pickle.load(f)

elif flag0 == 'test':
    with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/X/x_test.pickle".format(flag), "rb") as f:
        X_test = pickle.load(f)

    with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/data/data_test.pickle".format(flag), "rb") as f:
        D_test = pickle.load(f)

    with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/label_mat_test/label_mat_test{}.pickle".format(flag), "rb") as f:
        L_test = pickle.load(f)

    with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/labels/labels_test{}.pickle".format(flag), "rb") as f:
        Y_test = pickle.load(f)


def convert_sparse_matrix_to_sparse_tensor(matrix):
    coo = matrix.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

X = X.float()
X_test = X_test.float()

"""
print(X.shape)
print(Y.shape)
print(L.shape)
print(len(D))
print(type(X))
print(type(Y))
print(type(L))
print(type(D))
print("train",np.unique(Y, return_counts=True))
print("{}".format(flag0),np.unique(Y_test, return_counts=True))
"""

dclf = dummy.DummyClassifier()
dclf.fit(X, Y)
Y_baseline = dclf.predict(X_test)
#score = dclf.score(X_test, Y_test)
#print("Baseline:",score)

Xs, Ys, Ls, Ds = X, Y, L, D

Xs, Ys, Ls, Ds = split_data(X, Y, L, D, splits=[0.8, 0.2], stratify_by=Y, seed=123)


Xs.append(X_test)
Ys.append(Y_test)
Ls.append(L_test)
Ds.append(D_test)

print(lf_summary(Ls[1],Y=Ys[1]))

balance = sorted(Counter(Y_test).items())
balance2 = Counter(Y_test).values()

new_balance = []
for elem in balance:
    new_balance.append(elem[1]/sum(balance2))
print(sorted(Counter(Y_test).items()))
print(balance)
print(new_balance)

label_model = LabelModel(k=2, seed=123)
label_model.train_model(Ls[0], class_balance=new_balance,  n_epochs=500, log_train_every=50)

score = label_model.score((Ls[1], Ys[1]))

print('Trained Label Model Metrics:')
scores = label_model.score((Ls[1], Ys[1]), metric=['accuracy', 'precision', 'recall', 'f1'])


mv = MajorityLabelVoter(seed=123)
print('Majority Label Voter Metrics:')
scores = mv.score((Ls[1], Ys[1]), metric=['accuracy', 'precision', 'recall', 'f1'])

Y_train_ps = label_model.predict_proba(Ls[0])

Y_dev_p = label_model.predict(Ls[1])



"""
mv2 = MajorityClassVoter()
mv2.train_model(np.asarray(new_balance))
"""

#=np.asarray(new_balance))

#Y_baseline = mv2.predict(Ls[2])
pickling_on2 = open("ar_baseline_{}{}".format(flag0, flag), "wb")
pickle.dump(Y_baseline, pickling_on2)
print(Y_baseline)

# baseline majority:
"""
print(L_test)
Y_baseline = []
for row in L_test:
    print(row)
    Y_baseline.append(Counter(row).most_common()[0][0])
print(np.asarray(Y_baseline))
"""

Y_tes = label_model.predict(Ls[2])
pickling_on = open("ar_{}{}".format(flag0, flag), "wb")
print(Y_tes,type(Y_tes))
pickle.dump(Y_tes, pickling_on)

cm = confusion_matrix(Ys[1], Y_dev_p)





try:
    from metal.contrib.visualization.analysis import (
        plot_predictions_histogram,
        plot_probabilities_histogram,
    )

    plot_predictions_histogram(Y_tes, Ys[2], title="Label Distribution")

    Y_dev_ps = label_model.predict_proba(Ls[2])
    plot_probabilities_histogram(Y_dev_ps[:,0], title="Probablistic Label Distribution")

except ModuleNotFoundError:
    print("The tools in contrib/visualization/ require matplotlib. Try `conda/pip install matplotlib`.")

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
end_model = EndModel([1000, 10, 2], seed=123, device=device)
#print(type(Xs[0]))
#print(Xs[0])
#print(type(Y_train_ps))
#end_model.train_model((Xs[0], Y_train_ps), valid_data=(Xs[1], Ys[1]), lr=0.01, l2=0.01, batch_size=256, n_epochs=5, checkpoint_metric='accuracy', checkpoint_metric_mode='max')
#Xs = tf.convert_to_tensor(Xs)
#Y_train_ps = tf.convert_to_tensor(Y_train_ps)
#Ys = tf.convert_to_tensor(Ys)
#Ls = tf.convert_to_tensor(Ls)
end_model.train_model((Xs[0], Y_train_ps), valid_data=(Xs[1], Ys[1]), lr=0.01, l2=0.01, batch_size=256, n_epochs=5, checkpoint_metric='accuracy', checkpoint_metric_mode='max')


print("Label Model:")
score = label_model.score((Ls[2], Ys[2]), metric=['accuracy','precision', 'recall', 'f1'])

print()

print("End Model:")
score = end_model.score((Xs[2], Ys[2]), metric=['accuracy','precision', 'recall', 'f1'])
