#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:15:01 2019

@author: hosseintohidi
"""

from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
import pprint
from scipy.stats import wasserstein_distance
import networkx as nx
import time
import myEnv


def Gplot(G,s):
    pos = nx.get_node_attributes(G,'pos')
    node_labels = nx.get_node_attributes(G,'name')
    nx.draw(G, pos = pos, node_size = 15) #drawing nodes
    # offset labels:
    for key in pos.keys():
        pos[key] = (pos[key][0],pos[key][1]+ 0.015)
    nx.draw_networkx_labels(G,pos = pos, labels = node_labels,font_size=8)
    plt.title(f'Matching for s = {s}')
    plt.show()    


def gurobi_assignment(true_ws, num_arms, num_covs, N, timeLimit, plot = False, Gplot = False):
    I, J = np.arange(N), np.arange(N) # patients indices
    P, Q = np.arange(num_arms), np.arange(num_arms) # arms indices
    S = np.arange(num_covs) #covariates indices
    D = np.zeros([N,N,num_covs])
    for cov in S:
        for i in I:
            for j in J:
                D[i,j,cov] = abs(true_ws[i][0][cov]-true_ws[j][0][cov])
        
    print('GUROBI STARTS')
    tic = time.time()
    
    model = Model('Assignment problem using Wasserstein Distance')
    x = model.addVars(I,P, name='X', vtype= GRB.BINARY)
    d = model.addVars(P,Q,S, name='d', vtype= GRB.CONTINUOUS)
    z = model.addVars(I,J,P,Q,S, name='Z', vtype= GRB.BINARY)
    #constraints
    model.addConstrs((d[p,q,s] >= quicksum(D[i,j,s]*z[i,j,p,q,s] for i in I for j in J) for p in P for q in Q for s in S if p!=q ),'c0')
    model.addConstrs((x[i,p]+x[j,q] >= 2 * z[i,j,p,q,s] for i in I for j in J for p in P for q in Q for s in S),'c1')
    model.addConstrs(((num_arms-1) * x[i,p] == quicksum(z[i,j,p,q,s] for j in J for q in Q if i!=j and p!=q) for i in I for p in P for s in S),'c2')
    model.addConstrs((quicksum(x[i,p] for p in P) == 1 for i in I),'c3')
    model.addConstrs((quicksum(x[i,p] for i in I) == N//num_arms for p in P),'c4')
    model.addConstrs((quicksum(z[i,j,p,q,s] for j in J if i!=j ) <= 1 for i in I for p in P for q in Q for s in S if q!=p ),'c5')
    model.addConstrs((z[i,j,p,q,s] == z[j,i,q,p,s] for i in I for j in J for p in P for q in Q for s in S if q!=p),'c6')
    model.setObjective(1.0/(2*N//num_arms) * quicksum(d[p,q,s] for p in P for q in Q for s in S)  , GRB.MINIMIZE)
    model.update()
    model.write('treatment.lp')
    model.Params.timeLimit =  timeLimit
    model.optimize()
    sol = {}
    for v in model.getVars():
            if v.x !=0:
                sol[v.Varname] = v.x
                
    #pprint.pprint(sol)
    toc = time.time()
    print('time for MODEL0:', toc-tic)
    A = []
    for patient in range(N):
        A.append([sol.get('X['+ str(patient) +','+str(arm) +']',0) for arm in range(num_arms)])
            
    tot_dist = myEnv.find_wd(true_ws,A, plot= plot, figure_name='Gurobi_result')        

    if Gplot:
        for s in S:
            G = nx.Graph()
            X = [[] for i in range(M)]
            
            for key in sol.keys():
                if 'X' in key:
                    X[int(key[2:-1].split(',')[1])].append(int(key[2:-1].split(',')[0]))
            
            for i in range(len(X)):
                for j in range(len(X[0])):
                    G.add_node(X[i][j], pos=(i,j), name = 'P' + str(X[i][j]))
            for key in sol.keys():
                if 'Z' in key and int(key[-2])==s:
                    G.add_edge(int(key[2:-1].split(',')[0]), int(key[2:-1].split(',')[1])) 
            plt.subplot(len(S),1,s+1) 
            Gplot(G,s)

    return A, tot_dist











