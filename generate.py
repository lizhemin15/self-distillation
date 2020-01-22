# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:30:40 2020

@author: x1c
"""

import pickle
import numpy as np

file_path = 'C:/Users/x1c/.keras/datasets/cifar-10-batches-py/'
train_list = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
for name in train_list:
    with open(file_path+name, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        for i in range(len(data['labels'])):
            if np.random.random()<0.4:
                data['labels'][i]=int(np.mod(data['labels'][i]+1,10))

        
    with open(file_path+'new_'+name, 'wb') as file_pi:
        pickle.dump(data, file_pi)