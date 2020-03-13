#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:48:03 2019

@author: zihan
"""

import numpy as np
import sys
import math

train_name = sys.argv[1]
test_name = sys.argv[2]
ker = sys.argv[3]
spread = float(sys.argv[4])

train = np.loadtxt(train_name,delimiter =",")
test = np.loadtxt(test_name,delimiter =",")

#the setting of data on my computer
#train = np.loadtxt('Concrete_Data_RNorm_Class_train.txt', delimiter = ','  )
#test = np.loadtxt('Concrete_Data_RNorm_Class_test.txt', delimiter = ',')

y_train = train[: , 8]
X_train = np.c_[np.ones_like(y_train), train[:, 0:8]]

y_test = test[: , 8]
X_test = np.c_[np.ones_like(y_test), test[:, 0:8]]

def kernel(X1, ker, spread):
    K = []
    for i in X1:
        for j in X1:
            if ker == 'linear':
                K.append(np.dot(i, j))
            elif ker == 'quadratic':
                K.append((np.dot(i, j) + 1)**2)
            elif ker == 'gaussian':
                K.append(math.exp(- np.linalg.norm(i - j)**2 / (2* spread**2)))
    return np.array(K).reshape(len(X1),len(X1))

# I ran each kernel in my own computer
#K_linear = kernel(X_train, 'linear', 0)
#K_quad = kernel(X_train, 'quadratic', 0)
#K_gaus = kernel(X_train, 'gaussian', 0.05)
K = kernel(X_train, ker, spread)

def c(K, y):
    I = np.eye(len(K))
    c = np.dot(np.linalg.inv(K + 0.01*I), y)
    return c
#c_linear = c(K_linear_rev, y_train)
#c_quad = c(K_quad_rev, y_train)
#c_gaus = c(K_gaus_rev, y_train)
c = c(K, y_train)

def y(X1, X2, c, ker, spread):
    Y_hat_ker = []
    for a in X2:
        y = 0
        for b in range(len(X1)):
            if ker == 'linear':
                y += c[b]*(np.dot(X1[b], a.T))
            elif ker == 'quadratic':
                y += c[b]*((np.dot(X1[b], a.T) +1)**2)
            elif ker == 'gaussian':
                y += c[b]*(np.exp(- np.linalg.norm(X1[b] - a)**2 / (2* spread**2)))
        Y_hat_ker.append(y)
    return Y_hat_ker

#Y_hat_ker_linear = y(X_train, X_test, c_linear, 'linear', 0)
#Y_hat_ker_quad = y(X_train, X_test, c_quad, 'quadratic', 0)
#Y_hat_ker_gaus = y(X_train, X_test, c_gaus, 'gaussian', 0.05)
Y_hat_kernel = y(X_train, X_test, c, ker, spread)

def classify (data):
    Y_hat_kernel = []
    for y in data:
        if y >=  0.5:
            a = 1
        else:
            a = 0
        Y_hat_kernel.append(a)
    return Y_hat_kernel
#y_linear = classify(Y_hat_ker_linear)
#y_quad = classify(Y_hat_ker_quad)
#y_gaus = classify(Y_hat_ker_gaus)

y = classify(Y_hat_kernel)
 
def accuracy(Y, Y_hat):
    count = 0
    for i in range(len(Y)):
        if Y[i] == Y_hat[i]:
            count += 1
    accuracy = count / len(Y)
    return accuracy

#acc_linear = accuracy(y_test, y_linear)
#acc_quad = accuracy(y_test, y_quad)
#acc_gaus = accuracy(y_test, y_gaus)
acc_kernel = accuracy(y_test, y)
print ('the accuracy value on test data is ', acc_kernel)













