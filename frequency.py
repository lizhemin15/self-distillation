# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 09:47:17 2020

@author: x1c
"""

import numpy as np

def compute_distances_no_loops(Y, X): 
    #Input shape:Y,X:[N_samples,N_features],Output shape:dists:[N_samples,N_samples]
    dists = -2 * np.dot(X, Y.T) + np.sum(Y**2,axis=1) + np.sum(X**2, axis=1)[:, np.newaxis] 
    #print(type(dists))
    return dists

def normal_kernel(diff_x2,filter_wid): 
    #Input shape:diff_x2:in fact is the output of compute_distance_no_loops:[N_samples,N_samples],filter_wid=$\delta$
    #Output shape: have same shape with input
    gau_x2=np.exp(-diff_x2/2/filter_wid) 
    n_con=np.sum(gau_x2,axis=1,keepdims=True)
    n_gau_x2=gau_x2/n_con
    return n_gau_x2

def gauss_filter_normalize2(f_orig,n_gau_x2): 
    f_new=np.matmul(n_gau_x2,f_orig) 
    return f_new

def get_f_high_low(yy,xx,s_filter_wid,diff_x2=[]): 
    if len(diff_x2)==0: 
        diff_x2=compute_distances_no_loops(xx,xx) 
        #print(diff_x2)
    n_gau_x2_all=[]
    for filter_wid in s_filter_wid:
        n_gau_x2=normal_kernel(diff_x2,filter_wid) 
        n_gau_x2_all.append(n_gau_x2)  
    f_low=[] 
    f_high=[] 
    for filter_wid_ind in range(len(s_filter_wid)):
        f_new_norm=gauss_filter_normalize2(yy,n_gau_x2_all[filter_wid_ind]) 
        f_low.append(f_new_norm)
        f_high_tmp=yy-f_new_norm 
        f_high.append(f_high_tmp)
    return f_low, f_high

def cifar(x,y,delta=[1,10,20,50,100,200,1000]):
    x=x.reshape(-1,32*32*3)
    f_low, f_high=get_f_high_low(yy=y,xx=x,s_filter_wid=delta,diff_x2=[])
    delta_list=[]
    for i in range(len(f_low)):
        delta_now = np.sum(np.sum(f_low[i]**2))/(np.sum(np.sum(f_low[i]**2))+np.sum(np.sum(f_high[i]**2)))
        delta_list.append(delta_now)
        
    return delta_list

if __name__=='__main__':
    x=np.random.random((100,32*32*3))
    y=np.random.random((100,10))
    delta_list=cifar(x,y)
    
    