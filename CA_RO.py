# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:25:56 2020

@author: atohidi
"""

from gurobipy import *
import random
import numpy as np
from numpy.linalg import norm
from scipy.linalg import sqrtm    
import myEnv
import copy

def check_constraints(A, x_t, k):
    A = copy.deepcopy(A)
    A.append(x_t)
    total = np.array(A).sum(axis = 0)
    return all(i<=k for i in total) and np.array(x_t).sum() == 1
    
    
def CA_RO(true_ws, num_arms, N, plot = False):
    True_ws = copy.deepcopy(true_ws)
    true_ws = np.array(true_ws)[:,0,:]
    num_covs = len(true_ws[0])
    k = N//num_arms #number of patients in each arm
    S = np.arange(num_covs)
    P = np.arange(num_arms)
    Q = np.arange(num_arms)
    rho = 6 
    A = []
    
    # first 10 assignments are fully random
    for t in range(10):
        idx = np.random.randint(num_arms)
        A.append([1 if i==idx else 0 for i in range(num_arms)])
    
    # then RO
    for t in range(10, N):        
        # new x from greedy search
        sol = [] 
        Gamma = 0.5+ 3.5*np.random.random() #page 118
        if t>= N - 10:
            Gamma = 0 # page 116

        Gamma_tilda = Gamma**2 * (N-t) * num_covs 
    
        for ii in range(num_arms):
            x_t = [1 if idx == ii else 0 for idx in range(num_arms)]
            if check_constraints(A, x_t, k):   
                w_bar_t = 1/t * true_ws[:t,:].sum(axis = 0)
                Sigma_t = np.cov(true_ws[:t].T) 
                sqrt_Sigma_t = sqrtm(Sigma_t) # sqrt of negative number
                    
                
                z_list = []
                for p in range(num_arms):
                    for q in range(p,num_arms):
                        tmp_list=[]
                        for s in S:
                            mu1 =  sum([(true_ws[i,s]-w_bar_t[s])*(A[i][p]-A[i][q]) for i in np.arange(t-1)])+ \
                                   (true_ws[t,s]-w_bar_t[s])*(x_t[p]-x_t[q])+\
                                   np.sqrt(Gamma_tilda) * norm(sqrt_Sigma_t[s]) *\
                                   np.sqrt(2*k - sum([A[i][p] + A[i][q] for i in np.arange(t-1)]) -(x_t[p]+x_t[q]))
                                    
                
                            mu2 = sum([(true_ws[i,s]-w_bar_t[s])*(A[i][q]-A[i][p]) for i in np.arange(t-1)])+ \
                                  (true_ws[t,s]-w_bar_t[s])*(x_t[q]-x_t[p])+\
                                  np.sqrt(Gamma_tilda) * norm(sqrt_Sigma_t[s]) *\
                                  np.sqrt(2*k - sum([A[i][p] + A[i][q] for i in np.arange(t-1)]) -(x_t[p]+x_t[q]))
                                    
                            mu = 1/k * max(mu1,mu2)     
                            
                            
                            var1 = sum([(true_ws[i,s]-w_bar_t[s])**2 *(A[i][p]-A[i][q]) for i in np.arange(t-1)])+ \
                                   (true_ws[t,s]-w_bar_t[s])**2 *(x_t[p]-x_t[q])+\
                                   Gamma_tilda * norm(sqrt_Sigma_t[s])**2 *\
                                   int(k - sum([A[i][p] for i in np.arange(t-1)]) -x_t[p] >= 1)
                                  
                            var2 = sum([(true_ws[i,s]-w_bar_t[s])**2 *(A[i][q]-A[i][p]) for i in np.arange(t-1)])+ \
                                   (true_ws[t,s]-w_bar_t[s])**2 *(x_t[q]-x_t[p])+\
                                   Gamma_tilda * norm(sqrt_Sigma_t[s])**2 *\
                                   int(k - sum([A[i][q] for i in np.arange(t-1)]) -x_t[q] >= 1)
                                                
                            var = 1/k * max(var1, var2)
                            tmp_list.append(mu+rho*var)
                        
                        tmp = sum(tmp_list)
                        z_list.append(tmp)
                obj = max(z_list)
                                                                        
#                model = Model('CA-RO')
#                
#                z = model.addVar(name='Z', vtype = GRB.CONTINUOUS)
#                M = model.addVars(P, Q, S,name='M', vtype = GRB.CONTINUOUS)
#                V = model.addVars(P, Q, S,name='M', vtype = GRB.CONTINUOUS)
#                
#                model.addConstrs((z>= quicksum(M[p,q,s] + rho * V[p,q,s] for s in S) for p in P for q in Q if p < q),'c1')
#                
#                model.addConstrs((k*M[p,q,s] >= sum([(true_ws[i,s]-w_bar_t[s])*(A[i][p]-A[i][q]) for i in np.arange(t-1)])+ \
#                                                (true_ws[t,s]-w_bar_t[s])*(x_t[p]-x_t[q])+\
#                                                np.sqrt(Gamma_tilda) * norm(sqrt_Sigma_t[s]) *\
#                                                np.sqrt(2*k - sum([A[i][p] + A[i][q] for i in np.arange(t-1)]) -(x_t[p]+x_t[q]))\
#                                                for p in P for q in Q for s in S if p < q),'c2')
#                
#                model.addConstrs((k*M[p,q,s] >= sum([(true_ws[i,s]-w_bar_t[s])*(A[i][q]-A[i][p]) for i in np.arange(t-1)])+ \
#                                                (true_ws[t,s]-w_bar_t[s])*(x_t[q]-x_t[p])+\
#                                                np.sqrt(Gamma_tilda) * norm(sqrt_Sigma_t[s]) *\
#                                                np.sqrt(2*k - sum([A[i][p] + A[i][q] for i in np.arange(t-1)]) -(x_t[p]+x_t[q]))\
#                                                for p in P for q in Q for s in S if p < q),'c3')
#                
#                
#                model.addConstrs((k*V[p,q,s] >= sum([(true_ws[i,s]-w_bar_t[s])**2 *(A[i][p]-A[i][q]) for i in np.arange(t-1)])+ \
#                                                (true_ws[t,s]-w_bar_t[s])**2 *(x_t[p]-x_t[q])+\
#                                                Gamma_tilda * norm(sqrt_Sigma_t[s])**2 *\
#                                                int(k - sum([A[i][p] for i in np.arange(t-1)]) -x_t[p] >= 1)\
#                                                for p in P for q in Q for s in S if p < q),'c4')
#                                  
#                model.addConstrs((k*V[p,q,s] >= sum([(true_ws[i,s]-w_bar_t[s])**2 *(A[i][q]-A[i][p]) for i in np.arange(t-1)])+ \
#                                                (true_ws[t,s]-w_bar_t[s])**2 *(x_t[q]-x_t[p])+\
#                                                Gamma_tilda * norm(sqrt_Sigma_t[s])**2 *\
#                                                int(k - sum([A[i][q] for i in np.arange(t-1)]) -x_t[q] >= 1)\
#                                                for p in P for q in Q for s in S if p < q),'c5')
#                        
#                model.setObjective(z , GRB.MINIMIZE)
#                model.update()
#                model.setParam('OutputFlag', 0) #turn off output to console
#                #model.setParam("TimeLimit", TL)
#                #model.setParam("MIPGap", 0.06)
#                model.write('CA_RO.lp')
#                model.optimize()
#                obj2 = model.ObjVal
#                print(obj, obj2/obj, obj==obj2)
            else:
                obj = np.inf
            
            sol.append(obj)
            
        best_idx, best_obj = np.argmin(sol), np.min(sol)
        best_x = [1 if j== best_idx else 0 for j in range(len(sol))]
        A.append(best_x)
    
    
    reward = myEnv.find_wd(True_ws, A, plot= plot, figure_name = 'CA_RO')
    return A, reward
