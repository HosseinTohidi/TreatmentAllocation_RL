# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:40:02 2020

@author: atohidi
"""

import numpy as np
import myEnv

def sequential_assignment(true_ws,num_arms, N,plot = False):
    A = np.zeros([N, num_arms])
    for i in range(N):
        A[i, i % num_arms] = 1
    reward = myEnv.find_wd(true_ws, A.tolist(), plot = plot,figure_name='sequential_result')

    return A, reward