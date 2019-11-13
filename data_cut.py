import os
import numpy as np
import pickle
import _pickle as cPickle
from tqdm import tqdm
split_length = 512
num_start = 100000
num_end = 110000


if __name__ == "__main__":
    path = './dataset'
    des = './test/'
    items = os.listdir(path)
    for i in tqdm(range(len(items))):
        itemfile = os.path.join(path,items[i])
        data = np.fromfile(itemfile,dtype=np.int16)
        newdata = data[num_start*split_length:num_end*split_length]
        cPickle.dump(newdata,open(des+items[i],"wb"))

        