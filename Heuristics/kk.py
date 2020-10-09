# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 01:01:31 2020

@author: atohidi
"""
import numpy as np
from scipy.stats import f
from scipy.spatial import distance
import myEnv
import copy


def kk_assignment(true_ws, num_arms, N, plot = False):
    if num_arms != 2:
        raise Exception ('KK Assignment only works with two arm')
    
    N_thr = N // 10
    True_ws = copy.deepcopy(true_ws)
    true_ws = (np.array(true_ws)[:,0,:]).tolist()
    num_covs = len(true_ws[0])

    
    Lambda = 0.1 
    A, reservoir = [], []
    for t in range(N):
        if t <= N_thr or reservoir == []:
            rnd = np.random.random() <= 0.50
            if rnd:
                A.append([1,0])
            else:
                A.append([0,1])
            reservoir.append((t,true_ws[t]))
          #  print("here", t)
        else:
            # compute s_inverse
            S = np.cov(np.array(true_ws[:t]).T)
            S_inv = np.linalg.inv(S)
            # compute T^2
            F_star = f.ppf(Lambda, num_covs, t-num_covs)
            T_sq_star = num_covs * (t-1) / (t-num_covs) * F_star
            #compute distance of x_new (x_t) to all others in reservoir
            #all_dist = [distance.mahalanobis(true_ws[t], reservoir[r][1], S_inv) for r in range(len(reservoir))]
            all_dist =[]
            for r in range(len(reservoir)):
                v1, v2 = np.array(true_ws[t]), np.array(reservoir[r][1])
                all_dist.append(1/2*np.dot(np.dot((v1-v2).T, S_inv), (v1-v2)))
            T_sq_r_star, r_star = np.min(all_dist),np.argmin(all_dist)    
            if T_sq_r_star <= T_sq_star:
                pop = reservoir.pop(r_star)
                idx_1 = A[pop[0]].index(1)
                if idx_1 == 0:
                    A.append([0,1])
                else:
                    A.append([1,0])
              #  print(f'match found({t, pop[0]})')
            else:
#                print('****, WARNING', t, len(reservoir))
                #random assignment
                rnd = np.random.random() <= 0.50
                if rnd:
                    A.append([1,0])
                else:
                    A.append([0,1])
                reservoir.append((t,true_ws[t]))
    
    reward = myEnv.find_wd(True_ws, A, plot= plot, figure_name = 'kk')
    return A, reward
    
    
    


