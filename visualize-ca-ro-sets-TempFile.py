# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:56:17 2020

@author: atohidi
"""

batchNum = 1

batch = np.array(True_WS)[:,batchNum,:]

dff = pd.DataFrame(batch)

dff.plot()