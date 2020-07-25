# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:05:40 2020

@author: atohidi
"""
import numpy as np
def pocock(trueW,n,m,A,thresholds, prob = 3/4):
    """
    Python implementation of the code presented in https://github.com/nkorolko/Online_Clinical_Trials

	This function calculates assignments of n-th patient according to POCOCK BCD method
    It returns vector of assignments x_{np}, for p=1..m
    
    n - time stage
    trueW - vector of patients` properties (only the first n components will be used)
    m - number of the groups (m=2 always)
    A - Nxm matrix of previous assignemnts (we will use only the first n-1 rows)
    thresholds - sorted vector of size J-1 with points that split R into J levels
    thresholds = [-2,-1,0,1,2]
    prob, parameter p in Pocock and Simon model, 3/4 by default
    prob = 3/4
    We assume that 
    1) trueW are 1-dimensional i.i.d. standard N(0,10)
    2) 6 levels of covariate stratification with thresholds: thresholds=[-2, -1, 0, 1, 2] (J=6)
    
    Precalculate coefficients \nu
    
    nu=zeros(m) # aka nu_p^{n-1}, sizes of groups at the end of time stage n-1
    for p=1:m
        nu[p]=sum(A[1:n-1,p])
    end
    
    J - Number of levels for covariate
    """
    J = len(thresholds) + 1

    NN = np.zeros([J,2]) # Number of elements in level j=1...J and group m=1,2.
    
    for i in range(n-1):
        for j in range(J-1):
            if trueW[i] < thresholds[j]:
                if A[i][0] > 0.9:
                    NN[j,0]+=1
                else:
                    NN[j,1] += 1
                break
        if trueW[i] >= thresholds[J-2]:
            if A[i][0] > 0.9:
                NN[J-1,0] += 1
            else:
                NN[J-1,1] += 1

    # Current level NN_cur, level of covariate trueW[n]
    NN_cur = 1
    for j in range(J-1):
        if trueW[n-1] < thresholds[j]:
            NN_cur = j
            break
        NN_cur = J-1

    # Difference D_i(n)
    D = NN[NN_cur,0] - NN[NN_cur,1]
    
    phi = 0 # Baised Coin Probability
    if D == 0 :
        phi = 0.5
    elif D < 0:
        phi = prob
    else:
        phi = 1-prob
    
    y = np.zeros(m) # binary assignment variables
    
    x = np.random.rand() # toss a coin
    if x < phi :
        y[0] = 1
    else:
        y[1] = 1

    return list(y)

if __name__ == '__main__':
    thresholds = [-2, -1, 0, 1, 2]
    n = 2
    m = 2
    prob = 3/4 # default
    # previous assignments
#    A = [[1,0],
#         [1,0],
#         [1,0],
#         [1,0],
#         [1,0]]
    trueW = [1,-1,-2,3,1,1,-1,-2,3,1]
#    print(pocock(trueW,n,m,A,thresholds, prob))

    N = 10
    for n in range(N):
        if n == 0:
            A = [[1,0]]
        else:
            print(A)
            assign = pocock(trueW,n,m,A,thresholds, prob)
            A.append(assign)
            
        