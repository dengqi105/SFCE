# -*- coding:utf-8 -*-
# @FileName  :SFCE-HIT.py
# @Time      :2024/2/29/18:45
# @Author    :dengqi


import numpy as np
import os
import time
from Models.SFCE import SFCE

# %% parameters setting
print("****" * 10, "parameters setting", "****" * 10)
fs = 25000  # sampling frequency
sensor_index = [0, 1, 2, 3, 4, 5]
K = 7
L = 512  # sampele length
kernel_size = 45  # 45 is OK


print("K:", K)
print("L:", L)
print("S:", len(sensor_index))
print("kernel_size:", kernel_size)
print("****" * 25)

# %% LOAT DATA
path = "./dataset"
path_list = os.listdir(path)

shapes = []
all_data = []
for i in range(len(path_list)):
    datas = np.load(os.path.join(path, path_list[i]))
    # random_index = np.random.choice(datas.shape[2] - L, 1)[0].astype(int)
    # dataset = datas[:, sensor_index, random_index:random_index+L]
    data = datas[:, sensor_index, :L]
    print(data.shape)
    shapes.append(data.shape[0])
    all_data.append(data)
all_data = np.row_stack(all_data)

# %%

label = [0, 0, 1, 1, 2]
shape_list = [shapes[0] + shapes[1], shapes[2] + shapes[3], shapes[4]]

def get_label(label):
    labels = []
    for i in range(len(label)):
        labels = labels +  shapes[i]*[i]
    labels = np.array(labels)
    return labels


labels = get_label(label)

corrs = ["Correntropy", "Euclidean", "Covariance", "Correlation", "Minkowski", "Cosine"]  # correlation methods
d_methods = ["FDM", "EWT", "VMD"]  # dataset agumentation methods
clf_names = ["RR", "NB", "LR", "LDA", "LSVM", "GMSVM"]  # classifiers

corr = corrs[0]
d_method = d_methods[0]
clf = clf_names[0]

print("correlation method:", corr)
print("dataset agumentation method:", d_method)
print("classifier:", clf)

# %% LOAD MODEL
start_time = time.time()
Extractor = SFCE(fs=fs, K=K, kernel_size=kernel_size, clf=clf, corr=corr, d_method=d_method, z_score=True)
corr_f, corr_f1, corr_f2 = Extractor.transform(all_data)
end_time = time.time()
print("time:", end_time - start_time)

# %% TRAIN AND TEST

train_number = 13  # for each class
repeat_num = 20

print("K", K)
print("kernel_size:", kernel_size)
print("correlation method:", corr)
print("estimator:", clf)
print("train_number:", train_number)

Acc = np.zeros(repeat_num)
Acc1 = np.zeros(repeat_num)
Acc2 = np.zeros(repeat_num)
for i in range(repeat_num):

    train_id = []
    shape_num = 0
    for j in range(len(shape_list)):
        train_index = np.random.choice(shape_list[j], train_number, replace=False).astype(int)
        train_id.append(train_index + shape_num)
        shape_num += shape_list[j]
    train_id = list(np.row_stack(train_id).reshape((1, -1)).squeeze())

    test_id = list(set(list(np.arange(all_data.shape[0]))) - set(train_id))

    train_label = labels[train_id]
    test_label = labels[test_id]

    train_f, test_f = corr_f[train_id, :], corr_f
    train_f1, test_f1 = corr_f1[train_id, :], corr_f1
    train_f2, test_f2 = corr_f2[train_id, :], corr_f2

    predicted_label = Extractor.fit_predict(train_f, train_label, test_f)
    predicted_label1 = Extractor.fit_predict(train_f1, train_label, test_f1)
    predicted_label2 = Extractor.fit_predict(train_f2, train_label, test_f2)

    Acc[i] = np.sum(test_label == predicted_label[test_id]) / len(test_label) * 100
    Acc1[i] = np.sum(test_label == predicted_label1[test_id]) / len(test_label) * 100
    Acc2[i] = np.sum(test_label == predicted_label2[test_id]) / len(test_label) * 100

print('Acc:{}'.format(np.round(np.mean(Acc), 2)), 'Std:{}'.format(np.round(np.std(Acc), 2)))
print('Inner_Acc:{}'.format(np.round(np.mean(Acc1), 2)), 'Std:{}'.format(np.round(np.std(Acc1), 2)))
print('Intra_Acc:{}'.format(np.round(np.mean(Acc2), 2)), 'Std:{}'.format(np.round(np.std(Acc2), 2)))
