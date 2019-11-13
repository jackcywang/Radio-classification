#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 22:43:39 2019

@author: wangng
"""

import os 
import numpy as np
from dataset import Dataset
from config import *
from utils import *
from model import ResNet50
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
plt.switch_backend('agg')
os.environ['CUDA_VISIBLE_DEVICES']='1'

LABELS=['0','1','2','3','4','5','6','7','8','9','10','11','12',
'13','14','15','16','17','18','19','20','21','22','23','24',
'25','26','27','28','29']

checkpoint = 'weights/model.44-0.0000-1.0000.h5'

testdata= Dataset(path=test_path,split_length=split_length)
X_test,labels = testdata.get_data()
n_samples= X_test.shape[0]
test_idx = list(set(range(0,n_samples)))
X_test = X_test[:,:,np.newaxis]
Y_test = one_hot(list(map(lambda x: labels[x],test_idx)))
in_shp = list(X_test.shape[1:])
model = ResNet50(in_shp,classes=)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights(checkpoint)
score = model.evaluate(X_test,Y_test,batch_size=128)
print('evaluate score:', score)

predicts = model.predict(X_test,batch_size=128)

pre_labels=[]
true_labels=[]
for item in predicts:
    tmp = np.argmax(item, 0)
    pre_labels.append(tmp)
for item in Y_test:
    tmp = np.argmax(item, 0)
    true_labels.append(tmp)
conf_matrix = confusion_matrix(true_lables,pre_lables)
plot_confusion_matrix(conf_matrix,labels=LABELS)
plt.savefig(result_path+'/confusion matrix')

oa = accuracy_score(true_labels, pre_labels)
print('oa all:',oa)