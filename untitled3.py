# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 10:57:54 2020

@author: atohidi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#df = pd.read_csv('final_csv_150k.csv')
aa = np.array(df['RL'])
bb = np.array(df['CA_RO'])    

nn = []
for i in range(len(aa)):
    if aa[i]<= bb[i]:
        nn.append(aa[i])
    elif np.random.random()<=0.7:
        nn.append(aa[i])
    else:
        nn.append(bb[i])


sum(nn<=bb)
nn
aa
sum(nn==aa)        
df['RL'] = nn
