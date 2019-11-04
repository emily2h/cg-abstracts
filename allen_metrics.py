from sklearn import metrics
import numpy as np
import pickle
import sys

flag = sys.argv[1]
flag2 = sys.argv[2]

vci_external_test_Y = []
vci_external_test_X_words = []

filename = ''
if flag == 'test':
    filename = '/home/emily2h/Summer/cg-abstracts/vci_1543_abs_tit_key_apr_1_2019_test.csv'
if flag == 'true_test':
    filename = '/home/emily2h/Summer/cg-abstracts/vci_358_abs_tit_key_may_7_2019_true_test.csv'

with open(filename) as f:
    for line in f:
        if line.strip() != "":
            x_str, y_str = line.strip().split('\t')
            vci_external_test_X_words.append(x_str)
            y = [0] * 5
            for d in y_str.split():
                y[int(d)] = 1
            vci_external_test_Y.append(y)

ys = np.array(vci_external_test_Y)
labels = []
if flag2 == 'baseline':
    labels = pickle.load(open("/home/emily2h/Summer/metal/tutorials/label_mat_baseline_{}.pickle".format(flag), 'rb'))
else:
    labels = pickle.load(open("/home/emily2h/Summer/metal/tutorials/label_mat_{}.pickle".format(flag), 'rb'))

print(metrics.classification_report(ys, labels, digits=3))
print("Exact match: ", metrics.accuracy_score(ys, labels))

accu = np.array([metrics.accuracy_score(ys[:, i], labels[:, i]) for i in range(5)],
                        dtype='float32')

auc = np.array([metrics.roc_auc_score(ys[:, i], labels[:, i]) for i in range(5)],
                        dtype='float32')
print("Accuracy per label: ", accu)

print("Average accuracy: ", np.mean(accu))

print("AUC:", auc)

