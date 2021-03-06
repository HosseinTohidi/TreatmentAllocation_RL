# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:33:02 2020

@author: atohidi
"""
import numpy as np
from scipy.stats import wasserstein_distance as wd
from itertools import combinations 
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm
from scipy.linalg import sqrtm    

def extend_state(state, true_ws, thresholds):
    extended_state = []
    for batch in range(len(true_ws)):
        flat_state = list(state[batch].flatten())
        flat_state.extend(find_strata(true_ws[batch], thresholds))
        extended_state.append(flat_state)
    return extended_state


def new_arrivals(batch_size, thresholds):
    """
    return the covariates of new arrival (for each batch): 
           true_ws -> a list of size batch_size X len(threshold)     
    """
    true_ws = []
    for batch in range(batch_size):
        true_ws.append([np.random.choice(thresholds[cov],1)[0] for cov in range(len(thresholds))])
    return true_ws

def new_arrivals2(batch_size, thresholds, N, True_WS= []):
    """
    return the covariates of new arrival (for each batch): 
           true_ws -> a list of size batch_size X len(threshold)     
    """
    t = len(True_WS)
    S = len(thresholds)
    
    Gamma = 0.5+ 3.5*np.random.random() #page 118

    if t>= N - 10:
        Gamma = 0 # page 116
        
    if len(True_WS) <=10:
        return(new_arrivals(batch_size, thresholds))
    else:
        new_true_ws = []
        for batch in range(batch_size):
            true_batch = np.array(True_WS)[:,batch,:]    
            w_bar_t = 1/t * true_batch.sum(axis = 0)
            Sigma_t = np.cov(true_batch.T) 
            sqrt_Sigma_t = sqrtm(Sigma_t) # sqrt of negative number
            epsilon_norm = Gamma * np.sqrt((N-t)*S)
        
            dim = (N-t)*S # or any positive integer
            if dim != 0:
                rnd = epsilon_norm* np.random.random()
                eps = np.random.normal(size=(1,dim)) 
                eps /= np.linalg.norm(eps, axis=1)[:, np.newaxis]
                eps *= rnd
            else:
                eps = np.array([0 for i in range(S)])
            w_tilde = w_bar_t + np.dot(sqrt_Sigma_t, eps[0][:S])
            
            maxList = [0 for i in range(S)]
            minList = [np.inf for i in range(S)]
            for i in range(S):
                max_val = w_bar_t + np.dot(sqrt_Sigma_t, np.array([epsilon_norm  if ii == i else 0 for ii in range(S)]))
                for j in range(S):
                    if maxList[j] < max_val[j]:
                        maxList[j] = max_val[j]                
                min_val = w_bar_t + np.dot(sqrt_Sigma_t, np.array([-epsilon_norm if ii == i else 0 for ii in range(S)]))
                for j in range(S):
                    if minList[j] > min_val[j]:
                        minList[j] = min_val[j] 
            maxList, minList = np.array(maxList), np.array(minList)    
            if Gamma != 0:
                w_tilde = (w_tilde-minList)/(maxList-minList) 
                
            new_true_ws.append(w_tilde)
            
    return new_true_ws


def add_randomness(True_WS, batch_size, thresholds):
    S = len(thresholds)
    for batch in range(batch_size):
        true_batch = np.array(True_WS)[:,batch,:] 
        for s in range(S):
            max_ = true_batch.max(axis = 0)[s]
            min_ = true_batch.min(axis = 0)[s]
            if  max_==min_:
                print('Warning********************************************')
                idx = thresholds[s].index(max_)
                len_ = len(thresholds[s])
                possible_set = set(range(len_)) - set({idx})
                True_WS[-1][batch][s] = np.random.choice(list(possible_set))
    return True_WS
                


def new_arrivals3(batch_size, thresholds, N):
    True_WS = []
    for t in range(N):
        S = len(thresholds)
    
        Gamma = 0.5+ 3.5*np.random.random() #page 118
        if t>= N - 10:
            Gamma = 0 # page 116
        if len(True_WS) < 10:
            True_WS.append(new_arrivals(batch_size, thresholds))
            #return(new_arrivals(batch_size, thresholds))
        else:
            #modify True_WS if all equal to same number:
            if len(True_WS) == 10:
                True_WS = add_randomness(True_WS, batch_size, thresholds)
                    
            new_true_ws = []
            for batch in range(batch_size):
                true_batch = np.array(True_WS)[:,batch,:]    
                w_bar_t = 1/t * true_batch.sum(axis = 0)
                Sigma_t = np.cov(true_batch.T) 
                sqrt_Sigma_t = sqrtm(Sigma_t) # sqrt of negative number
                epsilon_norm = Gamma * np.sqrt((N-t)*S)
        
                dim = (N-t)*S # or any positive integer
                if dim != 0:
                    rnd = epsilon_norm* np.random.random()
                    eps = np.random.normal(size=(1,dim)) 
                    eps /= np.linalg.norm(eps, axis=1)[:, np.newaxis]
                    eps *= rnd
                else:
                    eps = np.array([0 for i in range(S)])
                w_tilde = w_bar_t + np.dot(sqrt_Sigma_t, eps[0][:S])                    
                new_true_ws.append(w_tilde)
            True_WS.append(new_true_ws)
    
    #modify True_ws to fit in 0 and 1 range
    tmp = []
    for batch in range(batch_size):
        True_WS_batch = np.array(True_WS)[:,batch,:] 
        max_vector = True_WS_batch.max(axis = 0)  
        min_vector = True_WS_batch.min(axis = 0)  
        std_vector = True_WS_batch.std(axis = 0)  
        range_ = max_vector - min_vector
        if 0 in range_:
            print(max_vector)
            print(min_vector)
            raise Exception ("Error: devide by 0")
        tmp.append((True_WS_batch - min_vector)/(max_vector - min_vector))
    final_ws = []
    for t in range(N):
        t_list = []
        for batch in range(batch_size):
            t_list.append([tmp[batch][t][j] for j in range(S)])
        final_ws.append(t_list)
        
    return final_ws



#                maxList = [0 for i in range(S)]
#                minList = [np.inf for i in range(S)]
#                for i in range(S):
#                    max_val = w_bar_t + np.dot(sqrt_Sigma_t, np.array([epsilon_norm  if ii == i else 0 for ii in range(S)]))
#                    for j in range(S):
#                        if maxList[j] < max_val[j]:
#                            maxList[j] = max_val[j]                
#                    min_val = w_bar_t + np.dot(sqrt_Sigma_t, np.array([-epsilon_norm if ii == i else 0 for ii in range(S)]))
#                    for j in range(S):
#                        if minList[j] > min_val[j]:
#                            minList[j] = min_val[j] 
#                maxList, minList = np.array(maxList), np.array(minList)    
#                if Gamma != 0:
#                    w_tilde = (w_tilde-minList)/(maxList-minList) 


def find_distance(*argv):
    total_distance = 0
    for i1,i2 in combinations(np.arange(len(argv[0])),2):
        if len(argv[0][i1]) ==0 or len(argv[0][i2])==0:        
            total_distance += 100 #np.inf
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
                 thresholds,
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
        self.thresholds = thresholds
        self.terminal_signal = terminal_signal

    def reset(self):
        self.state = np.zeros([self.batch_size, self.num_strata, self.num_arms])
        self.clock = 0
        self.assignment = [{(cov,arm):[] for (cov, arm) in list(itertools.product(np.arange(self.num_covs), np.arange(self.num_arms)))} for batch in range(self.batch_size)]
        self.terminal_signal = False
        
        return self.state     
            
    def find_reward(self):
        if self.clock < self.max_time:
           return np.zeros(self.batch_size)
        else:
           temp_reward = 50* np.array([reward_helper(self.assignment[batch], self.num_covs, self.num_arms) for batch in range(self.batch_size)])
#           temp_reward += np.array([self.state[batch].sum(axis =0).max() - self.state[batch].sum(axis =0).min() for batch in range(self.batch_size)])/self.num_covs
           temp_reward += 1* np.array([abs(self.state[batch].sum(axis =0) -  self.num_covs*self.max_time/ self.num_arms).sum() for batch in range(self.batch_size)])/self.num_covs

           return -1 * temp_reward #
        
    def step(self, actions, true_ws): # actions : 1 X batch_size 
        #translate actions to num_strata X 
        for batch in range(self.batch_size):
            #idx = actions[batch].index(1)
            self.state[batch,:,actions[batch]] += np.array(find_strata(true_ws[batch],self.thresholds))
            
        # update the clock and check for the end of the season
        self.clock += 1
        if self.clock >= self.max_time:
            self.terminal_signal = True
        
        # update the asssignment
        for batch in range(self.batch_size):
            for cov in range(self.num_covs):
                self.assignment[batch][(cov, actions[batch])].append(true_ws[batch][cov])    
        
        
            
        return self.state, self.find_reward(), self.terminal_signal


def myPlot(A, trueW, arms, figure_name="myPlt"):
    plt.close()
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
            sns.distplot(cov_values, hist=False,color= pltColors[arm])
    plt.tight_layout()
    plt.savefig(f'{figure_name}.PNG')




def find_wd(true_ws, A, plot= False, figure_name='myPlt'):
    num_arms = len(A[0])
    num_covs = len(true_ws[0][0])
    sol = [[] for i in range(num_arms)]
    for i in range(len(A)):
        idx = A[i].index(1)
        sol[idx].append(true_ws[i][0])
    total_dist = 0
    for cov in range(num_covs):
        total_dist += find_distance([np.array(sol[arm])[:,cov] for arm in range(num_arms)])
    #total_dist += np.array(A).sum(axis = 0).max() - np.array(A).sum(axis = 0).min()
    total_dist = 50* total_dist + abs(np.array(A).sum(axis = 0) - len(A)/num_arms).sum()
    if plot:
        myPlot(A, np.array(true_ws)[:,0,:], num_arms, figure_name)  
    return total_dist









# state represents the number of patients in each strata (|J_1|+|J_2|+...|J_covs|) in aech arm  


if __name__ == '__main__':        
    arms = 2
    N = 100
    thresholds = [[0,1],
                  [5,10,15,20,40],
                  [10,20,30],
                  [-5,-3,-1,0,1,3,5]] 
    num_strata= np.sum([len(thresholds[i]) for i in range(len(thresholds))])
    batch_size = 2
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
    
    
    
    ### note: I changed the def of actions from 0,1,0,... to the index of 1 in each batch
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