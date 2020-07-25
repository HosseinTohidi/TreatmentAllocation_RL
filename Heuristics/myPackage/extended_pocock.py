# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:05:40 2020

@author: atohidi
"""
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt

def extended_pocock(trueW,arms,A,thresholds):
    """
    N: arms k = 0,1, ..., N
    M: factors i = 0, 1, ..., M
    threshold (n_i): levels for each factors
    x_ijk: number of patients in arm k with level n_i 
    r_i: New patient attributes
    
    
    Python implementation of the code presented in https://github.com/nkorolko/Online_Clinical_Trials

	This function calculates assignments of n-th patient according to POCOCK BCD method
    It returns vector of assignments x_{np}, for p=1..m
    
    n - time stage
    trueW - matrix of patients` properties (only the first n components will be used)
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
    def find_imbalance(threshold, true_W):  # true_w and thresholds are for one covariate
        J = len(threshold)
        n = len(A) # already allocated patients
        NN = np.zeros([J ,arms]) # Number of elements in level j=1...J and group arm=1,2,3...|arms|.   
        
        for person in range(n):
            group = A[person].index(1)
            for j in range(J):
                if true_W[person] <= threshold[j]:
                    NN[j][group] += 1
                    break
        # level of current patient in one covariate
        NN_cur = 0
        for j in range(J):
            if true_W[-1] <= threshold[j]:
                NN_cur = j
                break
        
        # NN[NN_cur,m]+=1 for differet m value, we need only one row of NN
        one_row = copy.deepcopy(NN[NN_cur])
        std_list = []
        for arm in range(arms):
            one_row[arm] += 1
            std_list.append(np.std(one_row))
            one_row[arm]-=1   
        return std_list
    
    num_covariates = len(trueW[0])
    total_imbalance = []
    for cov in range(num_covariates):
        threshold = thresholds[cov]
        true_W = np.array(trueW)[:,cov]
        total_imbalance.append(find_imbalance(threshold, true_W))
        
    sum_imbalance_allCov = np.array(total_imbalance).sum(axis=0) # we can instead weighted sum them (here weights are all one.)
    # now we design a biased coin (prob for each arm)
    t = 3/4
    prob = 1/(arms-t )*(1-t*sum_imbalance_allCov/sum(sum_imbalance_allCov))
    # normalize it
    prob_norm = prob/sum(prob)
    print(prob_norm)
    # generate a uniform random number
    cumsum_prob = np.cumsum(prob_norm)
    # random genertion 
    rnd = np.random.random()
    idx = len(cumsum_prob[cumsum_prob>=rnd])-1
    assign = [1 if arm==idx else 0 for arm in range(arms)]
    print(assign)
    return assign

def myPlot(A, trueW, arms):
    sol = {}
    for arm in range(arms):
        sol[arm] = []
    for i in range(len(A)):
        idx = A[i].index(1)
        sol[idx].append(trueW[i])
    covs = len(trueW[0])    
    pltColors = ['g','k','b','r','y']
    for cov in range(covs):
        plt.subplot(covs,1,cov+1)
        for arm in range(arms):
            cov_values = np.array(sol[arm])[:,cov]
            sns.distplot(cov_values, hist=False,color= pltColors[arm], label= str(arm))
if __name__ == '__main__':
    arms = 3
    N = 100
    thresholds = [[0,1],
                  [5,10,15,20,40],
                  [10,20,30],
                  [-5,-3,-1,0,1,3,5]]    
    prob = 3/4 # default
    trueW = []        
    for i in range(N):
        trueW.append([np.random.randint(0,2), 
                      np.random.randint(0,40),
                      np.random.randint(0,30),
                      np.random.randint(-5,5)])
        
    
    
    for n in range(N):
        if n == 0:
            A = [[1 if i==0 else 0 for i in range(arms)]]
        else:
           # print(A)
            assign = extended_pocock(trueW[:n+1],arms,A,thresholds)
            A.append(assign)
    myPlot(A,trueW,arms)    
        