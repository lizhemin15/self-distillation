# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 08:41:18 2020

@author: x1c
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('historynew.txt', 'rb') as file_pi:
    history=pickle.load(file_pi)
    
loss_name=['loss','val_loss','dis_loss','dis_val_loss']
acc_name=['acc','val_acc','dis_acc','dis_val_acc']
fig = plt.gcf()
for name in loss_name:
    plt.plot(history[name],label=name)
plt.legend()

plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
plt.savefig('loss.png')

fig = plt.gcf()
for name in acc_name:
    plt.plot(history[name],label=name)
plt.plot(np.array(history['dis_val_acc'])-np.array(history['val_acc']),label='$\Delta$val_acc')
plt.legend()

plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
plt.savefig('acc.png')


fre=history['fre']
dis_fre=history['dis_fre']
plt_num=5
plt_fre=[]
plt_dis=[]
for i in range(len(fre)):
    plt_fre.append(fre[i][plt_num])
    plt_dis.append(dis_fre[i][plt_num])
    
fig = plt.gcf()
plt_fre = np.array(plt_fre)
plt_dis = np.array(plt_dis)
plt.plot(plt_fre,label='fre')
plt.plot(plt_dis,label='dis_fre')
plt.plot(plt_fre-plt_dis,label='$\Delta$')
plt.legend()

plt.xlabel('epoch')
plt.ylabel('frequency')
plt.show()
plt.savefig('frequency.png')