#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:53:26 2019

@author: zihan
"""

Import numpy as np
import random
import sys

train_name = sys.argv[1]
test_name = sys.argv[2]
esp = float(sys.argv[3])
eta = float(sys.argv[4])

train = np.loadtxt(train_name,delimiter =",")
test = np.loadtxt(test_name,delimiter =",")

#the setting of data on my computer
#train = np.loadtxt('Concrete_Data_RNorm_Class_train.txt', delimiter = ','  )
#test = np.loadtxt('Concrete_Data_RNorm_Class_test.txt', delimiter = ',')

y_train = train[: , 8]
X_train = np.c_[np.ones_like(y_train), train[: ,0:8]]

y_test = test[: , 8]
X_test = np.c_[np.ones_like(y_test), test[:, 0:8]]

def sigmoid(X):
    return 1.0/(1+np.exp(-X))

def SGA(data, y, esp, eta):
    w = np.zeros(len(data[1, :]))
    t = 0 
    while True:
        index = np.arange(len(data))
        random.shuffle(index)
        w0 = w.copy()
        for i in index:
            gradient =  (y[i] - sigmoid(np.dot(w.T, data[i, :])))* data[i, :]
            w += eta * gradient
        w1 = w.copy()
        t += 1
        if  np.linalg.norm(w1 - w0) < esp :
            break
    return (w1, t)

w = SGA(X_train, y_train, esp, eta)[0]
t= SGA(X_train, y_train, esp, eta)[1]

print('w value is ' , w)

def classify (data):
    Y_hat = []
    for i in range(len(data)):
        prob1 = sigmoid(np.dot(w.T, data[i]))
        if prob1 >=  0.5:
            y_hat = 1
        else:
            y_hat = 0
        Y_hat.append(y_hat)
    return Y_hat

Y_hat_train = classify(X_train)
   

def accuracy(Y, Y_hat):
    count = 0
    for i in range(len(Y)):
        if Y[i] == Y_hat[i]:
            count += 1
    accuracy = count / len(Y)
    return accuracy

acc_train = accuracy (y_train, Y_hat_train)
print('the accuracy for train database is ', acc_train)

Y_hat_test = classify(X_test)
acc_test = accuracy(y_test, Y_hat_test)
print('the accuracy for test database is ', acc_test)




















































