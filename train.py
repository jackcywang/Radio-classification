#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 22:43:39 2019

@author: wangng
"""

import os
import sys
import numpy as np
from dataset import Dataset
from utils import *
from model import ResNet50
from keras.optimizers import adam
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
os.environ['CUDA_VISIBLE_DEVICES']='1'
from logger import Logger
import argparse

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--root_path', type=str, default='/media/lab1/E1/jaywang')
    parse.add_argument('--data_path', type=str, default='./trainval')
    parse.add_argument('--weight_dir', type=str, default='./weights')
    parse.add_argument('--model_name', type=str, default='/model.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5')
    parse.add_argument('--log_dir', type=str, default='./tensorboard')
    parse.add_argument('--result_path', type=str, default='./result')
    parse.add_argument('--split_rate', type=int, default=0.8)
    parse.add_argument('--split_length',type=int, default=512)
    parse.add_argument('--num_classes',type=int, default=30)
    parse.add_argument('--batch_size', type=int, default=512)
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('--seed', type=int, default=2020, help='random seed') # 随机种子
    return parse.parse_args()

args = parse_args()
#打印训练日志
sys.stdout = Logger('train.log', sys.stdout)
#获取数据和标签
dataset = Dataset(path=args.data_path,split_length=args.split_length) 
datas,labels = dataset.get_data()

#随机种子设成固定，保证每次读取的数据一致
np.random.seed(args.seed)
n_samples = datas.shape[0]
n_train = n_samples*args.split_rate
trian_idx = np.random.choice(range(0, n_samples), size = int(n_train), replace = False)
val_idx = list(set(range(0, n_samples)) - set(trian_idx))

X_train = datas[trian_idx]
X_val = datas[val_idx]
X_train = X_train[:,:,np.newaxis]
X_val = X_val[:,:,np.newaxis]
#将标签转成one-hot形式
trainy = list(map(lambda x: labels[x],trian_idx))
valy = list(map(lambda x: labels[x],val_idx))
Y_trian = one_hot(trainy)
Y_val = one_hot(valy)

#构建模型
in_shp = list(X_train.shape[1:])
print(X_train.shape,in_shp)
model = ResNet50(in_shp,classes=args.num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#记录训练过程，加入学习衰减，early stop策略
tensorboard = TensorBoard(log_dir=args.log_dir)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
modelcheck = ModelCheckpoint(args.weight_dir+args.model_name, monitor='val_loss',verbose=0,save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=0)
    
history = model.fit(X_train,Y_trian,batch_size=args.batch_size,
                epochs=args.epochs,validation_data=(X_val,Y_val),
                callbacks=[tensorboard, reduce_lr, modelcheck, early_stop])
#打印准确率和损失曲线
plot_acc_loss(args.result_path,history)