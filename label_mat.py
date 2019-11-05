import pickle
import numpy as np
import sys

flag = sys.argv[1]
flag2 = sys.argv[2]


if flag2 != 'baseline':
    flag2 = ''
if flag2 == 'baseline':
    flag2 = flag2 + '_'

def func(x):
    if x == 2:
        return 0
    return 1

with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/ar/ar_{}{}0".format(flag2, flag), "rb") as f:
    label0 = pickle.load(f)
with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/ar/ar_{}{}1".format(flag2, flag), "rb") as f:
    label1 = pickle.load(f)
with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/ar/ar_{}{}2".format(flag2, flag), "rb") as f:
    label2 = pickle.load(f)
with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/ar/ar_{}{}3".format(flag2, flag), "rb") as f:
    label3 = pickle.load(f)
with open("/home/emily2h/Summer/cg-abstracts/data_encompassing/ar/ar_{}{}4".format(flag2, flag), "rb") as f:
    label4 = pickle.load(f)


print(label0.shape,label1.shape,label2.shape,label3.shape,label4.shape)

mat = np.asmatrix([label0, label1, label2, label3, label4]).transpose()
print(mat)
matfunc = np.vectorize(func)
mat = matfunc(mat)
print(mat.shape)
print(type(mat))
print(mat)

pickling_on = open("data_encompassing/label_mat_{}/label_mat_{}{}.pickle".format(flag, flag2, flag), "wb")
pickle.dump(mat, pickling_on)
