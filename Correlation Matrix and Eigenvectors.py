#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:24:18 2019

@author: zihan
"""

import numpy as np
import matplotlib.pyplot as plt

airfoil_data = np.loadtxt('airfoil_self_noise.dat')

#Problem_a
#get the mean for each column vector
mean_list = []
for i in range(len(airfoil_data[0])):
        mean = airfoil_data[:,i].sum() / len(airfoil_data)
        mean_list.append(mean) 
print(mean_list)
#transfer type form list to array
mean_df = np.asarray(mean_list)

#get the total mean of each entry
average= sum(map(sum, airfoil_data))/(np.size(airfoil_data))
print(average)

#get the square of the difference between the total mean and each entry value
def square(x):
    return (x - average)**2
airfoil_data_vec = np.vectorize(square)
squared_result = square(airfoil_data)
#get the total variance
var = sum(map(sum, squared_result))/(np.size(airfoil_data))
print(var)


#Problem_b
#get the centered data matrix
mean_matrix= np.matrix([mean_df]*len(airfoil_data))
centered_data = airfoil_data - mean_matrix
#get the sample covariance matrix by inner products
cov_inner = np.dot(centered_data.T, centered_data) / len(airfoil_data)
print(cov_inner)

#get the same covariance matrix by outer products
M = 0
for j in range(len(centered_data)):
    Z = np.dot(centered_data[j,:].T, centered_data[j,:])
    M += Z
    
cov_outer = M / len(airfoil_data)
print(cov_outer)

#Problem_c
#calculate the correlation matrix using cosine between centered attribute vectors
corr_list = []
for i in range(len(airfoil_data[0])):
    L0 = np.sqrt(np.dot(centered_data[:,i].T, centered_data[:,i]))
    for j in range(len(airfoil_data[0])):
        L1 = np.sqrt(np.dot(centered_data[:,j].T, centered_data[:,j]))
        corr =  np.dot(centered_data[:,i].T, centered_data[:,j])/(L0*L1)
        corr_list.append(corr)
print(corr_list)
#transfer correlation list into a matrix
corr_matrix = np.reshape(corr_list, (6,6))
print(corr_matrix)

"""according to the correlation matrix, 
attribute2 and attribute5 are most correlated,
attribute2 and attribute3 are most anti-correlated,
attribute1 and attribute3 are least correlated.
"""
#draw the scatter plot of the correlation of each pair
pairs = [i for i in range(36)] 
plt.scatter(pairs, corr_matrix)
plt.xlabel('Pairs#')
plt.ylabel('Correlation')
plt.title('Correlation Scatter Plot')
plt.show()
            
#Problem_d
#get x0: two non-zero dimensional column vectors with unit length
n, d = cov_inner.shape
x0 = np.random.rand(d,2)
x0 = x0 / np.linalg.norm(x0)

#the iteration
def scalar(A):
    return float(np.dot(A[:,1].T, A[:,0])/np.dot(A[:,0].T, A[:,0]))

xi = []
while True:
    eigenvec_m= np.dot(cov_inner, x0 )
    #orthogonalize b column
    eigenvec_m[:,1] = eigenvec_m[:,1] - scalar(eigenvec_m)*eigenvec_m[:,0]
    x0 = eigenvec_m / np.linalg.norm(eigenvec_m, axis = 0)
    eigenvec1_m = np.dot(cov_inner, x0)
    eigenvec1_m[:,1] = eigenvec1_m[:,1] - scalar(eigenvec1_m)*eigenvec1_m[:,0]
    x0 = eigenvec1_m / np.linalg.norm(eigenvec1_m, axis = 0)
    #get the list of original data points xi
    xi.append(eigenvec1_m)
    #quit the iteration when the difference is smaller than epsilon
    if np.abs(eigenvec1_m[0,0] - eigenvec_m[0,0]) < 0.0001:
        break

#the first two eigenvectors
u1 = x0[:,0]
u2 = x0[:,1]
test = u1.T.dot(u2)

print(u1)
print(u2)
print(test)

#try to get the projection of xi on eigenvectors
airfoil_new = np.dot(x0.T, xi[0])










    
    





































