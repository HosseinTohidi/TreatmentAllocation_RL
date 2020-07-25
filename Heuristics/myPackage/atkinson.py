# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:09:09 2020

@author: atohidi
"""
import numpy as np
def atkinson(trueW, n, m, A, thresholds, prob):
    """
    Python implementation of the code presented in https://github.com/nkorolko/Online_Clinical_Trials
	This function calculates assignments of n-th patient according to SMITH BCD method [1984b]
    It returns vector of assignments x_{np}, for p=1..m
    
    n - patients (n-1 previously assigned patient plus a new patient)
    trueW - vector of patients` properties (only the first n components will be used)
    m - number of the groups(m=2 always)
    A - nxm matrix of previous assignemnts (we will use only the first n-1 rows)
    thresholds, prob will be discarded! 
    We assume that 
    1) trueW are 1-dimensional i.i.d. standard normal (S=1)
    """
    x=0
    for i in range(n-1):
        if A[i][0]>0.9:
            x += trueW[i]
        else:
            x -= trueW[i]
    
    x/=np.linalg.norm(trueW[1:n-1])**2
    x*=trueW[n-1]
        
    phi=(1-x)**2/((1-x)**2+(1+x)**2)
    
    y=np.zeros(m) # binary assignment variables
    
    xx=np.random.rand() # toss a coin
    if xx<phi:
        y[0]=1
    else:
        y[1]=1
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
    print(atkinson(trueW,n,m,A))
