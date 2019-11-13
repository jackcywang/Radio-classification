# -*- coding: utf-8 -*-
import os 
import numpy as np
from tqdm import tqdm
class Dataset():
    def __init__(self, path, split_length):
        self.path = path 
        self.split_length=split_length
    
    def label_index(self):
        label_dict={}
        datafiles = os.listdir(self.path)
        for file in datafiles:
            label_name = file.split('-')[-1].split('.')[0]
            label_index = int(file.split('-')[0])
            label_dict[label_index]=label_name
        return label_dict
            
    
    def get_data(self):
        Datas=[]
        Labels=[]
        datafiles = os.listdir(self.path)
        print('读取文件中...')
        for index,file in enumerate(datafiles):
            print('文件{}'.format(index+1))
            file_path = os.path.join(self.path, file)
            label_index = int(file.split('-')[0])
            print(file_path,label_index)
            datarray = np.fromfile(file_path,dtype=np.int16)
            arr_mean = np.mean(datarray)
            arr_std = np.std(datarray)
            data_norm = (datarray-arr_mean)/arr_std
            # for i in range(0,len(data_norm),self.split_length):
            for i in tqdm(range(int(data_norm.shape[0]/self.split_length))):
                sample = data_norm[i:i+self.split_length]
                Datas.append(sample)
                Labels.append(label_index)
        Datas = np.vstack(Datas)
        print(Datas.shape,len(Labels))
        print(Labels[-5:])
        print('文件读取结束！！！')
        return Datas, Labels


if __name__ == "__main__":
    data = Dataset(path='./trainval/',split_length=512)
    _, _ = data.get_data() 

         


    