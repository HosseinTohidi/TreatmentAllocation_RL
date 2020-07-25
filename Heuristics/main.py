# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:00:01 2020

@author: atohidi
"""
from scipy.stats import wasserstein_distance as wd
from myPackage import antognini, atkinson, pocock
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def Distance(A, trueW, plot= False):
    """
    compute wasserstein distance between vector m1, m2 
    m0: list of trueW of elements where A[element][0] == 1
    m1: list of trueW of elements where A[element][1] == 1
    """
    m0, m1 = [], []
    for i in range(len(A)):
        if A[i][0] == 1:
            m0.append(trueW[i])
        else:
            m1.append(trueW[i])
    if plot:
#        plt.subplot(2,1,1)
        sns.distplot(m0, hist=False)
#        plt.subplot(2,1,2)
        sns.distplot(m1,hist=False)    
    return wd(m0,m1)

def enroll(trueW,N,m,thresholds,prob,func):
    """
    Simulates the sequential enrollment of N patients. 
    It follows func for treatment allocation.
    func can be one of the following functions:
        1- antognini.antognini
        2- atkinson.atkinson
        3- pocock.pocock
    """
    for n in range(N):
        if n == 0:
            A = [[1,0]]
        else:
            #print(A)
            assign = func(trueW,n,m,A,thresholds,prob)
            A.append(assign)
    return A

# comparing different methods
thresholds = [-3, -2, -1, 0, 1, 2, 3]
m = 2
N = 1000
prob = 3/4 # default
trueW = np.sqrt(10) * np.random.randn(N) # Nromal distributed with mean of 0 and variance of 10 


plt.subplot(1,3,1)
A = enroll(trueW,N,m,thresholds,prob,antognini.antognini)
plt.title('antognini')
print("antognini: ", Distance(A, trueW, plot = True))

plt.subplot(1,3,2)
A = enroll(trueW,N,m,thresholds,prob,atkinson.atkinson)
plt.title('atkinson')
print("atkinson: ", Distance(A, trueW, plot = True))

plt.subplot(1,3,3)
A = enroll(trueW,N,m,thresholds,prob,pocock.pocock)
plt.title('pocock')
print("pocock: ", Distance(A, trueW, plot = True))


