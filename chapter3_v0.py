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
import myEnv


device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu', 0) #args.gpu_num)
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument("--use_gpu", type=bool, default=True)
parser.add_argument("--gpu_num", type=int, default=3)
parser.add_argument("--batchSize", type=int, default=2, help='number of episodes at each training step')
parser.add_argument('--actor_lr', type=float, default=0.001, help='entropy term coefficient (default: 0.01)')
parser.add_argument('--critic_lr', type=float, default=0.001, help='entropy term coefficient (default: 0.01)')
parser.add_argument('--method', type=str, default='fc', help='fc | att')
args, unknown = parser.parse_known_args()

batch_size = 32 #args.batchSize
num_arms = 3
N = 100
thresholds = [[0,1],
              [5,10,15,20,40],
              [10,20,30],
              [-5,-3,-1,0,1,3,5]] 
num_strata= np.sum([len(thresholds[i]) for i in range(len(thresholds))])

len_state = num_strata * num_arms 
len_extended_state = len_state + num_strata


env = myEnv.trialEnv(state = [],
                     assignment=[],
                     clock = 0,
                     batch_size=batch_size,
                     num_strata=num_strata,
                     num_covs= len(thresholds),
                     num_arms=num_arms,
                     max_time=N,
                     thresholds=thresholds,
                     terminal_signal = False
                     )
env.reset()




class Actor(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(len_extended_state, 256)
        self.affine2 = nn.Linear(256, 256)
        self.affine3 = nn.Linear(256, num_arms)
    def forward(self, x):
        ''' do the forward pass and return a probability over actions
        Input:
                x: state -> shape: batch_size X (num_stratra*num_arms)
        returns:
                prob: a probability distribution ->shape: batch_size X num_arms
        '''
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_scores = self.affine3(x)
        probs = F.softmax(action_scores, dim=1)

        return probs


class Critic(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(len_extended_state, 256)
        self.affine2 = nn.Linear(256, 1)

    def forward(self, x):
        ''' do the forward pass and return a probability over actions
        Input:
                x: state  -> shape: batch_size X 120
        returns:
                v: value of being at x -> shape: batch_size X 1 
        '''
        x = F.relu(self.affine1(x))
        v = self.affine2(x).squeeze()
        return v
    
# create actor and critic network
actor = Actor().to(device)
critic = Critic().to(device)

# create optimizers
actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)  
critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr)


def select_action(extended_state, variance=1, temp=1):
    # this function selects stochastic actions based on the policy probabilities 
    state = torch.tensor(extended_state, dtype=torch.float32, device=device).unsqueeze(0)
    probs = actor(state)
    m = Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    return action.to('cpu').numpy()[0], log_prob


# multiple rollout
def rollout(env):
    states, rewards, log_probs = [], [], []
    # play an episode
    state = env.reset()
    while True:  
        true_ws = myEnv.new_arrivals(batch_size, thresholds) # generate new arrivals
        extended_state = myEnv.extend_state(state, true_ws, thresholds)
        action, log_prob = select_action(extended_state) # select an action
        states.append(extended_state)
        log_probs.append(log_prob)
        state, reward, done = env.step(action,true_ws)
        rewards.append(reward)
        if done:
            break
    return states, rewards, log_probs


def train(states, rewards, log_probs):
    rewards_path, log_probs_paths, avg_reward_path = [], [], []
#    for batch in range(len(rewards)):
    R = 0
    P = 0
    for i in reversed(range(len(rewards))):
        R = rewards[i] + args.gamma * R
        rewards_path.insert(0, R)
        P = log_probs[i] + P
        log_probs_paths.insert(0, P)

    log_probs_paths = torch.stack(log_probs_paths)
    # rewards_path: np.array(|batch|X|episod|), each element is a reward value
    # log_probs_paths:np.array(|batch|X|episod|), each element is a tensor
    rewards_path = torch.tensor(rewards_path, dtype=torch.float32, device=device)
    states = torch.tensor(states, device=device) 

    value = critic(states).float()  #.view(-1, len_extended_state
   
    # take a backward step for actor
    actor_loss = -torch.mean(((rewards_path - value.detach()) * log_probs_paths)) #.squeeze()
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    # take a backward step for critic
    loss_fn = torch.nn.MSELoss()
    critic_loss = loss_fn(value, rewards_path)
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    result = {}
    result['rew'] = rewards_path.mean().item()
    result['actor_loss'] = actor_loss.item()
    result['critic_loss'] = critic_loss.item()
    result['value'] = torch.mean(value).item()

    return result


   
rws = []
torchMean = []

def run_training(budget):
    for i_episode in range(budget):
        states, rewards, log_probs = rollout(env)
        result = train(states, rewards, log_probs)
        rws.append(result['rew'])
        torchMean.append(result['value'])
        if i_episode % 20 == 0:
            print(i_episode, result)
        if i_episode % 100 == 0:
            print('actor norm:', torch.norm(torch.cat([i.flatten() for i in actor.parameters()])))


run_training(100)

import matplotlib.pyplot as plt 

plt.plot(-1*np.array(rws[2000:]), 'b-', label = 'Total # of Infection')
plt.plot(-1*np.array(torchMean[100:]), 'r-' ,label = 'critic value')
plt.xlabel('episodes')
plt.legend()


