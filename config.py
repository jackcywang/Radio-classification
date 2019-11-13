# -*- coding: utf-8 -*-
root_path = '/media/lab1/E1/jaywang'
data_path =root_path+'/trainval'
weights_dir = root_path+'/weights'
model_name= weights_dir+ '/model.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5'
log_dir = root_path+'/tensorboard'
result_path = root_path+'/result'
split_length = 512
num_classes = 30
batch_size = 512
epochs = 100