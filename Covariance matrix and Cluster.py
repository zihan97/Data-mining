#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:15:26 2019

@author: zihan
"""

import numpy as np
import math
import random
import sys

trainname = sys.argv[1]
k = int(sys.argv[2])
eps = float(sys.argv[3])

train = np.loadtxt(trainname, delimiter = ',')

if trainname == 'iris.txt':
    X = train[:, 0: len(train[0])-1]
    C = []
    for i in range(len(train)):
        if train[i, len(train[0])-1] == 'Iris-setosa':
            C.append(0)
        elif train[i, len(train[0])-1] == 'Iris-versicolor':
            C.appen(1)
        else:
            C.append(2)
else:
    X = train[:, 0: len(train[0])-1]
    C = train[:, len(train[0])-1].reshape(-1, 1)



def density(x, mu, cov, d):
    cov += 0.0001 * np.identity(d)
    cov_sqrt = math.sqrt(np.linalg.det(cov))
    cov_inv = np.linalg.inv(cov)
    x_center = x - mu
    dens = (1.0 / (np.power(2.0 * math.pi, d/2.0) * cov_sqrt))* math.exp((-0.5) * np.dot(np.dot(x_center.T, cov_inv), x_center))
    return dens


n = len(X)
d = len(X[0])

def em(X, k, eps):
    t = 0
    mu = np.zeros((k,d))
    for i in range(k):
        for a in range(d):
            mu[i][a] = random.uniform(-1, 1)
    cov = []
    for i in range(k):
        cov.append(np.identity(d))
    cov = np.array(cov)

    P = np.zeros(k)
    for i in range(k):
        P[i] = 1.0 / k

    w = np.zeros((k, n))
    while True:
        t += 1
        old_mu = np.copy(mu)
        for j in range(n):
            total = 0
            for a in range(k):
                total += density(X[j], mu[a], cov[a], d) * P[a]
            for i in range(k):
                w[i][j] = density(X[j], mu[i], cov[i], d) * P[i] / total

        for i in range(k):
            sum1 = 0
            sum2 = 0
            for j in range(n):
                sum1 += w[i][j] * X[j]
                sum2 += w[i][j]
            mu[i] = sum1 / sum2

            sum1 = 0
            for j in range(n):
                sum1 += w[i][j] * np.outer(X[j] - mu[i], np.transpose(X[j] - mu[i]))
            cov[i] = sum1 / sum2

            P[i] = sum2 / n

        error = 0
        for i in range(k):
            error += np.linalg.norm(mu[i] - old_mu[i]) ** 2
        if error <= eps:
            clusters = [-1] * n
            for j in range(n):
                cluster = -1
                max_prob = 0
                for i in range(k):
                    if w[i][j] > max_prob:
                        max_prob = w[i][j]
                        cluster = i
                clusters[j] = cluster
            sizes = []
            for i in range(k):
                size = clusters.count(i)
                sizes.append(size)
            break
            
    return (mu, cov, t, clusters, sizes, w)

result = em(X, k, eps)


K = len(np.unique(C))

pur = 0
for i in range(k):
    maxi = 0
    Ci = set()
    for a in range(n):
        if result[3][a] == i:
            Ci.add(a)
    
    for j in np.unique(C):
        Tj = set()
        for a in range(n):
            if C[a] == j:
                Tj.add(a)
        intersect = len(Ci.intersection(Tj))
        maxi = max(maxi, intersect)
        pur += maxi

    purity = pur / (n * K)

for i in range(len(result[0])):
    print("Mean for cluster", i+1, result[0][i])
print()
for i in range(len(result[1])):
    print("Covariance matrix for cluster", i+1, result[1][i])
print()
print("Number of iterations:", result[2])
print()
print("Final cluster assignment of all points: ", result[3])
print()
for i in range(len(result[4])):
    print("Size of cluster", i+1, result[4][i])
print()
print("Purity score: ", purity)