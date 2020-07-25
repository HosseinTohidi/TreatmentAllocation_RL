# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:46:36 2020

@author: atohidi
"""

import argparse
import copy
import pandas as pd
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
action_dict = defaultdict()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Multinomial
from env import trialEnv


device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu
                                   else 'cpu', 0) #args.gpu_num)
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--entropy_coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument("--use_gpu", type=bool, default=True)
parser.add_argument("--gpu_num", type=int, default=3)
parser.add_argument("--batchSize", type=int, default=2,
                    help='number of episodes at each training step')

parser.add_argument('--actor_lr', type=float, default=0.001,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--critic_lr', type=float, default=0.001,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--method', type=str, default='fc',
                    help='fc | att')
args, unknown = parser.parse_known_args()


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




class Actor(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(len_state, 256)
        self.affine2 = nn.Linear(256, 256)
        self.affine3 = nn.Linear(256, groups_num)

    def forward(self, x):
        ''' do the forward pass and return a probability over actions
        Input:
                x: state -> shape: batch_size X 120
        returns:
                prob: a probability distribution ->shape: batch_size X 20
        '''
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_scores = self.affine3(x)

        return action_scores


class Critic(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(len_state, 256)
        self.affine2 = nn.Linear(256, 1)

    def forward(self, x):
        ''' do the forward pass and return a probability over actions
        Input:
                x: state  -> shape: batch_size X 120
        returns:
                v: value of being at x -> shape: batch_size 
        '''
        x = F.relu(self.affine1(x))
        v = self.affine2(x).squeeze()
        return v


