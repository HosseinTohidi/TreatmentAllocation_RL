# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:33:02 2020

@author: atohidi
"""
import numpy as np
from scipy.stats import wasserstein_distance as wd
from itertools import combinations 
import itertools

def find_distance(*argv):
    total_distance = 0
    for i1,i2 in combinations(np.arange(len(argv[0])),2):
        if len(argv[0][i1]) ==0 or len(argv[0][i2])==0:        
            total_distance += np.inf
        else:
            total_distance += wd(argv[0][i1],argv[0][i2])
    return total_distance


def find_strata(true_w, thresholds):
    sol = []
    for cov in range(len(true_w)):
        val, threshold = true_w[cov], thresholds[cov]
        J = np.zeros(len(threshold))
        idx = sum(np.array(threshold) < val)
        J[idx] = 1
        sol.extend(list(J))
    return sol

def reward_helper(assign_dict, num_covs, num_arms): # for a particular batch {(cov,arm):[list of trueWs]}
    total_dist = 0
    for cov in range(num_covs):
        total_dist += find_distance([assign_dict[(cov,arm)] for arm in range(num_arms)])
    return total_dist        

class trialEnv(object):
    """
    =================================================================================================
    # Main Variables
    
    state       :  number of patient within each strata at each arm for each batch
                   state.shape = batch X num_strata X num_arms
    num_strata  :  total number of strata
                   num_strata = \sum_(cov in covs) J_(cov) 
    assignment  :  True covariate values for patients within each arm
                   - list of size batch_size of dictionaries with all (cov,arm) combinations
                     as keys and list of patients true covariate as the value
                     i.e. assignment = [{(cov = 0, arm = 0): [0, 1], (cov = 0, arm = 1): [1, 0]}] 
                     for two arms one covariate and batch_size = 1     
   =================================================================================================
    # Methods
    
   __init__: initialize the object
   reset   : reset the object by:
             - reseting the clock (number of patient) to zero 
             - initializing the state (all zero) 
             - assining empty list for all assignment dictionaries (for all batches)
   step    : a new patient is arrived and assigned to an arm. 
             update time, state, as well as assignment, and compute the reward
    =================================================================================================   
    """
    def __init__(self, 
                 state,
                 assignment,
                 clock,
                 batch_size,
                 num_strata,
                 num_covs,
                 num_arms,
                 max_time,
                 terminal_signal = False
                 ):
        self.state = state 
        self.assignment = assignment
        self.clock = clock
        self.batch_size = batch_size
        self.num_strata = num_strata
        self.num_covs = num_covs
        self.num_arms = num_arms
        self.max_time = max_time
        self.terminal_signal = terminal_signal

    def reset(self):
        self.state = np.zeros([batch_size, num_strata, arms])
        self.clock = 0
        self.assignment = [{(cov,arm):[] for (cov, arm) in list(itertools.product(np.arange(self.num_covs), np.arange(self.num_arms)))} for batch in range(self.batch_size)]

        return self.state     
            
    def find_reward(self):
        if self.clock < self.max_time:
           return np.zeros(self.batch_size)
        else:
           return np.array([reward_helper(self.assignment[batch], self.num_covs, self.num_arms) for batch in range(self.batch_size)])
        
    def step(self, actions, true_ws): # actions : batch_size X arms
        #translate actions to num_strata X 
        for batch in range(batch_size):
            idx = actions[batch].index(1)
            self.state[batch,:,idx] += np.array(find_strata(true_ws[batch],thresholds))
            
        # update the clock and check for the end of the season
        self.clock += 1
        if self.clock >= self.max_time:
            self.terminal_signal = True
        
        # update the asssignment
        for batch in range(self.batch_size):
            for cov in range(self.num_covs):
                self.assignment[batch][(cov, actions[batch].index(1))].append(true_ws[batch][cov])    
        
        
            
        return self.state, self.find_reward(), self.terminal_signal


# state represents the number of patients in each strata (|J_1|+|J_2|+...|J_covs|) in aech arm  


if __name__ == '__main__':        
    arms = 2
    N = 100
    thresholds = [[0,1],
                  [5,10,15,20,40],
                  [10,20,30],
                  [-5,-3,-1,0,1,3,5]] 
    num_strata= np.sum([len(thresholds[i]) for i in range(len(thresholds))])
    batch_size = 1
    #initial_state = np.zeros([batch_size, num_strata, arms]) 
    
    trueW = []        
    for i in range(N):
        trueW.append([np.random.randint(0,2), 
                      np.random.randint(0,40),
                      np.random.randint(0,30),
                      np.random.randint(-5,5)])
        
    env = trialEnv(state = [],
                   assignment=[],
                   clock = 0,
                   batch_size=batch_size,
                   num_strata=num_strata,
                   num_covs= len(thresholds),
                   num_arms=arms,
                   max_time=N,
                   terminal_signal = False
                   )
    env.reset()
    
    actions=[[1,0],[0,1],[1,0]]
    true_ws =[[1,13,25,0],[1,16,25,0],[0,16,25,0]]
    new_state, new_reward, terminal_signal = env.step(actions,true_ws)
    
    actions=[[0,1],[1,0],[0,1]]
    true_ws =[[0,5,12,3],[0,1,29,4],[1,15,25,-2]]
    new_state, new_reward, terminal_signal = env.step(actions,true_ws)
    
    actions=[[1,0],[0,1],[1,0]]
    true_ws =[[1,13,25,0],[1,16,25,0],[0,16,25,0]]
    new_state, new_reward, terminal_signal = env.step(actions,true_ws)
    
    actions=[[0,1],[1,0],[0,1]]
    true_ws =[[0,20,12,3],[0,1,29,4],[1,15,25,-2]]
    new_state, new_reward, terminal_signal = env.step(actions,true_ws)