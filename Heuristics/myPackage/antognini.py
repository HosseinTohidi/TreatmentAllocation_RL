# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:21:29 2020

@author: atohidi
"""
import numpy as np

def antognini(trueW,n,m,A,thresholds,prob):
    """
    Python implementation of the code presented in https://github.com/nkorolko/Online_Clinical_Trials
    
    This function calculates assignments of n-th patient according to C-ABCD(F^a) method
    It returns vector of assignments x_{np}, for p=1..m
    
    n - patients (n-1 previously assigned patient plus a new patient)
    trueW - vector of patients` properties (only the first n components will be used)
    m - number of the groups (always 2)
    A - nxm matrix of previous assignemnts (we will use only the first n-1 rows)
    thresholds - sorted vector of size J-1 with points that split R into J levels
    thresholds = [-2,-1,0,1,2]
    prob will be discarded. (not used)
    We assume that 
    1) trueW are 1-dimensional i.i.d. N(0,10)
    2) 6 levels of covariate stratification with thresholds: thresholds=[-2, -1, 0, 1, 2] (J=6)
    
    J - Number of levels for covariate
    """
    J = len(thresholds) + 1

    NN=np.zeros([J,2]) # Number of elements in level j=1...J and group m=1,2.
    
    for i in range(n-1):
        for j in range(J-1):
            if trueW[i]<thresholds[j]:
                if A[i][0]>0.9:
                    NN[j,0]+=1
                else:
                    NN[j,1]+=1
                break
        if trueW[i]>=thresholds[J-2]:
            if A[i][0]>0.9:
                NN[J-1,0]+=1
            else:
                NN[J-1,1]+=1
    # Current level NN_cur, level of covariate trueW[n]
    NN_cur=1
    for j in range(J-1):
        if trueW[n-1]<thresholds[j]: #n'th element true value
            NN_cur=j
            break
        NN_cur=J-1
    # Difference D_i(n)
    D = 0
    D = NN[NN_cur,0] - NN[NN_cur,1]
    
    a = J-1 # exponent in function F^a(D)
    phi=0 # Baised Coin Probability
    if D>=0 and D<=1:
        phi=0.5
    elif D>1:
        phi=(D**a+1)**(-1)
    else:
        phi=1-((-D)**a+1)**(-1)            
    y = np.zeros(m) # binary assignment variables
    x = np.random.rand() # toss a coin
    if x < phi:
        y[0] = 1
    else:
        y[1] = 1
    return list(y)

if __name__ == '__main__':
    thresholds = [-2, -1, 0, 1, 2]
    n= 5
    m = 2
    # previous assignments
    A = np.array([[1,0],
                  [1,0],
                  [1,0],
                  [1,0],
                  [1,0]])
    
    trueW = [1,-1,-2,3,1]
    print(antognini(trueW,n,m,A,thresholds))
    
    
    