#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 08:47:16 2019

@author: zihan
"""

import numpy as np
import sys
#the working directory is different on your computer
#import os
#os.chdir('/Users/zihan/Desktop/spider_wd')

train_name = sys.argv[1]
test_name = sys.argv[2]
alpha = float(sys.argv[3])

.y_train = train[: ,11]
x_train = train[: , 0:11]
y_test = test[: ,11]
x_test = test[: , 0:11]

#the function to get augmented data matrix
def augmentation(dataframe):
    x0 = np.ones(len(dataframe))
    return np.c_[x0, dataframe]

x_train_aug = augmentation(x_train)
x_test_aug = augmentation(x_test)

# the function of QR decomposition
def GramSchmidt(X):
    Q = np.ones_like(X)
    R = np.zeros((X.shape[1], X.shape[1]))
    count = 0
    for x in X.T:
        u = np.copy(x)
        for i in range(0, count):
            r=np.dot(Q[:,i].T, x) / np.linalg.norm (Q[: ,i]) **2
            R[i,count] = r
            u -= r * Q[:, i]
        Q[:, count] = u
        R[count,count] = 1
        count += 1
    return (Q, R)

Q_train = GramSchmidt(x_train_aug)[0]
R_train = GramSchmidt(x_train_aug)[1]

# the function to do backsolve
def back(dataframe, y):
    Q = GramSchmidt(dataframe)[0]
    R = GramSchmidt(dataframe)[1]
    N = np.dot(Q.T, Q)
    N_inv = np.linalg.inv(N)
    RHS = np.dot(N_inv, np.dot(Q.T, y))
    
    W = np.zeros((dataframe.shape[1],))
    number = dataframe.shape[1]-1
    for w in reversed(RHS):
        for j in range (number,dataframe.shape[1]-1):
            w -= R[number,j+1] * W[j+1]
        W[number] = w
        number -= 1
    return (W)

W_train = back(x_train_aug, y_train)

y_train_pred = np.dot(x_train_aug, W_train)
y_test_pred = np.dot(x_test_aug, W_train)

#calculate TSS, SSE, R_squared
y_train_mean= np.asarray([y_train.sum() / len(y_train)] * len(y_train))
y_train_centered = y_train - y_train_mean

y_test_mean= np.asarray([y_test.sum() / len(y_test)] * len(y_test))
y_test_centered = y_test - y_test_mean

SSE_train = np.linalg.norm(y_train - y_train_pred)**2
TSS_train = np.linalg.norm(y_train_centered)**2

SSE_test = np.linalg.norm(y_test - y_test_pred)**2
TSS_test = np.linalg.norm(y_test_centered)**2

R_squared_train = (TSS_train - SSE_train) / TSS_train
R_squared_test = (TSS_test - SSE_test) / TSS_test

print("Linear Regression: the W is ", W_train)
print("the L2 norm of W is ",np.linalg.norm(W_train))
print("the SSE of train is ", SSE_train)
print("the R_squared of train is",R_squared_train)
print("the SSE of test is ", SSE_test)
print("the R_squared of test is",R_squared_test)


#perform ridge regression to get W
def ridge(df, y, a):
    I = np.identity(df.shape[1])
    A_sqrt = I * np.sqrt(a)
    D_slice = np.r_[df, A_sqrt]
    zeros = np.zeros((df.shape[1], ))
    y_slice = np.r_[y, zeros]
    W_ridge = back(D_slice, y_slice)
    return (W_ridge)

#plan to test alpha = [0.1, 10, 250, 1000, 10000]

W_train_ridge = ridge(x_train_aug, y_train, alpha)
print ("Ridge Regression: the W_ridge is " , W_train_ridge)
print("the L2 norm of W_ridge is ",np.linalg.norm(W_train_ridge))
y_train_ridge_pred = np.dot(x_train_aug, W_train_ridge)
SSE_train_ridge = np.linalg.norm(y_train - y_train_ridge_pred)**2
R_squared_train_ridge = (TSS_train - SSE_train_ridge) / TSS_train
print("the SSE of train is " , SSE_train_ridge )
print("the R_squared of train is " , R_squared_train_ridge)
 
y_test_ridge_pred = np.dot(x_test_aug, W_train_ridge)
SSE_test_ridge = np.linalg.norm(y_test - y_test_ridge_pred)**2
R_squared_test_ridge = (TSS_test - SSE_test_ridge) / TSS_test
print("the SSE of test is " , SSE_test_ridge )
print("the R_squared of test is " , R_squared_test_ridge)
    
    
    


