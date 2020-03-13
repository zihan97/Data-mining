#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:38:50 2019

@author: zihan
"""

import numpy as np
import random
import sys

train_name = sys.argv[1]
test_name = sys.argv[2]
m = float(sys.argv[3])
n = float(sys.argv[4])
epochs = float(sys.argv[5])

m = int(m)
epochs = int(epochs)

#import os
#os.chdir('/Users/zihan/Desktop/spider_wd')

train = np.loadtxt(train_name, delimiter =",")
test = np.loadtxt(test_name, delimiter =",")

X_train = train[:, 0:9]
y_train = train[:, 9]

X_test = test[:, 0:9]
y_test = test[:, 9]

def one_hot (data):
    data_hot = np.array([1 if i == a else 0 for a in data for i in range(1,8)])
    return data_hot.reshape(len(data), 7)

def reLU(net_z):
    net_z[net_z < 0] = 0
    return net_z

y_train_hot = one_hot(y_train)

Wh = np.random.uniform(-0.1, 0.1, (len(X_train[0]), m))
bh = np.random.uniform(-0.1, 0.1, (m, 1))

Wo = np.random.uniform(-0.1, 0.1, (m, 7))
bo = np.random.uniform(-0.1, 0.1, (7, 1))

#MLP training
t = 0 

while t < epochs:
    index = np.arange(len(X_train))
    random.shuffle(index)
    for i in index:
        xi = X_train[i].reshape(-1,1)
        yi = y_train_hot[i].reshape(-1, 1)
        net_zi = np.dot(Wh.T, xi) + bh
        zi = reLU(net_zi)
        net_oi= np.dot(Wo.T, zi) + bo
        sum_oi = np.sum(np.exp(net_oi))
        oi = np.exp(net_oi) / sum_oi
        
        deriv_relu = np.zeros(zi.shape)
        deriv_relu[zi > 0] = 1.0
        
        net_gradient_o = oi - yi
        net_gradient_h = np.multiply(deriv_relu,np.dot(Wo,net_gradient_o))
        bo = bo - n * net_gradient_o
        bh = bh - n * net_gradient_h
        
        Wo = Wo - n * np.dot(zi, net_gradient_o.T)
        Wh = Wh - n * np.dot(xi, net_gradient_h.T)
    t = t + 1
    
def y_pred (Wh, Wo, bh, bo, data):
    y_pred = []
    for k in range(len(data)):
        x = data[k].reshape(-1,1)
        net_z = np.dot(Wh.T, x) + bh
        z = reLU(net_z)
        net_o = np.dot(Wo.T, z) + bo
        sum_ok = np.sum(np.exp(net_o))
        o = np.exp(net_o) / sum_ok
        y_pred.append(o)
    return np.array(y_pred).reshape(len(data), 7)

y_pred_train = y_pred(Wh, Wo, bh, bo, X_train)
y_pred_test = y_pred(Wh, Wo, bh, bo, X_test)

def accuracy (y_pred, y):
    correct = np.sum(np.argmax(y_pred, axis = 1 ) == (y-1))   
    return correct/ len(y_pred)

acc_train = accuracy(y_pred_train, y_train)
acc_test = accuracy(y_pred_test, y_test)


print("Weights of hidden layer:\n", Wh)
print("Weights of ouput layer:\n", Wo)
print("Bias of hidden layer:\n", bh)
print("Bias of output layer:\n", bo)
print("Train Accuracy:", acc_train)
print("Test Accuracy:", acc_test)
    





    
    
    
    
    
    
    

    
    






    
        
    