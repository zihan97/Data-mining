#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:02:19 2019

@author: zihan
"""

import numpy as np
import sys
import math

train_name = sys.argv[1]
test_name = sys.argv[2]
C = float(sys.argv[3])
eps = float(sys.argv[4])
ker = sys.argv[5]
spread = float(sys.argv[6])

train = np.loadtxt(train_name,delimiter =",")
test = np.loadtxt(test_name,delimiter =",")


y_train = train[: , 8]
X_train = np.c_[train[:, 0:8], np.ones_like(y_train)]

y_test = test[: , 8]
X_test = np.c_[test[:, 0:8], np.ones_like(y_test)]

def kernel(X, ker, spread):
    K = []
    for i in X:
        for j in X:
            if ker == 'linear':
                K.append(np.dot(i, j))
            elif ker == 'quadratic':
                K.append(np.dot(i, j)**2)
            elif ker == 'gaussian':
                K.append(math.exp(- np.linalg.norm(i - j)**2 / (2* spread**2)))
    return np.array(K).reshape(len(X),len(X))

def step(K):
    N = []
    for k in range(len(K)):
        N.append(1 / K[k, k])
    return N

N = step(kernel(X_train, ker , spread))

def SGA(X, y, C, esp, ker, spread, N):
    a = np.zeros(len(X))
    t = 0 
    index = np.arange(len(X))
    while True:
        a0 = a.copy()
        for k in index :
            add = 0
            for i in index:
                if ker == 'linear':
                    add += a[i]*y[i]*np.dot(X[i], X[k])
                elif ker == 'quadratic':
                    add += a[i]*y[i]*(np.dot(X[i], X[k])**2)
                elif ker == 'gaussian':
                    add += a[i]*y[i]*(np.exp(- np.linalg.norm(X[i] - X[k])**2 / (2* spread**2)))
            gradient =  1 - y[k]*add
            a[k] = a[k] + N[k] * gradient
            if a[k] < 0:
                a[k] = 0
            elif a[k] > C:
                a[k] = C
        a1 = a.copy()
        t += 1
        print(t)
        print(np.linalg.norm(a1 - a0))
        if  np.linalg.norm(a1 - a0) < esp :
            break
    return (a1)

A = SGA(X_train, y_train, C, eps, ker, spread, N)


ai= []
SV = []
SV_y = []
K = []
for k in range(len(A)):
    if A[k] > 0:
        K.append(k)
        ai.append(A[k])
        SV.append(X_train[k])
        SV_y.append(y_train[k])
print('The support-vector i are:', K)
print('The corresponding alphas are:', ai)


      
def predict(X, SV, SV_y, ai, ker, spread):
    Y = []
    for z in X:
        r = 0
        y = 0
        for i in range(len(SV)):
            if ker == 'linear':
                r += ai[i]* SV_y[i] *(np.dot(SV[i], z))
            elif ker == 'quadratic':
                r += ai[i]*SV_y[i]*(np.dot(SV[i], z)**2)
            elif ker == 'gaussian':
                r += ai[i]*SV_y[i]*(np.exp(- np.linalg.norm(SV[i] - z)**2 / (2* spread**2)))
        if r > 0:
            y = 1.0
        elif r <= 0:
            y = -1.0
        Y.append(y)
    return Y

Y_hat = predict(X_test, SV, SV_y, ai, ker, spread)


def accuracy(Y, Y_hat):
    count = 0
    for i in range(len(Y)):
        if Y[i] == Y_hat[i]:
            count += 1
    accuracy = count / len(Y)
    return accuracy

acc = accuracy(y_test, Y_hat)
print('The accuracy on test set is: ', acc)

def weigh(ai, SV, SV_y, ker):
    index = np.arange(len(SV))
    if ker == 'linear':
        w = np.zeros(len(SV[1]))
        for i in index:
            w += ai[i]*SV_y[i]*SV[i]
    elif ker == 'quadratic':
        w = np.zeros(37)
        for i in index:
            x_feature = []
            for j in range(len(SV[i])-1):
                phi_square = np.square(float(SV[i][j]))
                x_feature.append(phi_square)
                for q in range(j+1, len(SV[i])-1, 1):
                    phi_other = np.sqrt(2)*float(SV[i][j])*float(SV[i][q])
                    x_feature.append(phi_other)
            x_feature.append(1)
            w += np.multiply((ai[i]*SV_y[i]), x_feature)
    return w
    
w = weigh(ai, SV, SV_y, ker)
print('The weight vectors in feature space are: ',  w)

    
  














