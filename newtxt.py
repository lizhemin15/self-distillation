# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 08:23:03 2020

@author: x1c
"""

import pickle
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models generate history')
parser.add_argument('--num', default=1,
                    help='iteration num')
args = parser.parse_args()

history = {'loss':[],'acc':[],'val_loss':[],'val_acc':[],'fre':[],
           'dis_loss':[],'dis_acc':[],'dis_val_loss':[],'dis_val_acc':[],'dis_fre':[]}


name_space = ['no_dis_train','dis_train','no_dis_test','dis_test']
for i in range(int(args.num)):
    for name in name_space:
        with open(str(i)+name+'.txt', 'rb') as f:
            data = pickle.load(f)
        if name == 'dis_train':
            history['dis_loss'].append(data['train_loss'])
            history['dis_acc'].append(data['train_acc'])
        if name =='dis_test':
            history['dis_val_loss'].append(data['test_loss'])
            history['dis_val_acc'].append(data['test_acc'])
            history['dis_fre'].append(data['fre'])
            
        if name == 'no_dis_train':
            history['loss'].append(data['train_loss'])
            history['acc'].append(data['train_acc'])
        if name =='no_dis_test':
            history['val_loss'].append(data['test_loss'])
            history['val_acc'].append(data['test_acc'])
            history['fre'].append(data['fre'])

            
with open('historynew.txt', 'wb') as file_pi:
    pickle.dump(history, file_pi)



