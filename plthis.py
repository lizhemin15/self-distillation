# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 08:41:18 2020

@author: x1c
"""

import matplotlib.pyplot as plt
import pickle

with open('historynew.txt', 'rb') as file_pi:
    history=pickle.load(file_pi)
    
loss_name=['loss','val_loss','dis_loss','dis_val_loss']
acc_name=['acc','val_acc','dis_acc','dis_val_acc']

for name in loss_name:
    plt.plot(history[name],label=name)
plt.legend()
plt.show()

for name in acc_name:
    plt.plot(history[name],label=name)
plt.legend()
plt.show()

fre=history['fre']
dis_fre=history['dis_fre']
plt_num=5
plt_fre=[]
plt_dis=[]
for i in range(len(fre)):
    plt_fre.append(fre[i][plt_num])
    plt_dis.append(dis_fre[i][plt_num])
plt.plot(plt_fre,label='fre')
plt.plot(plt_dis,label='dis_fre')
plt.legend()
plt.show()
