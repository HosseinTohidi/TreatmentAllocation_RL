# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:46:36 2020

@author: atohidi
"""
import sys
import argparse
import copy
import pandas as pd
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
import seaborn as sns
import Heuristics.extended_pocock as pocock
import Gurobi_assignment as g_a
import random_assignment as r_a




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

batch_size = 30 #args.batchSize
num_arms = 3
N = 99
#thresholds = [[0,1],
#              [5,10,15,20,40],
#              [10,20,30],
#              [-5,-3,-1,0,1,3,5]] 
thresholds = [[0,1],
              [0,0.2,0.4,0.6,0.8,1],
              [0,0.5,1],
              [0,0.3,0.6,0.9,1]] 

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
state = env.reset()

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
    state = torch.tensor(extended_state, dtype=torch.float32, device=device)  #.unsqueeze(0)
    probs = actor(state)
    #print(probs)
    m = Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    # entropy = - torch.sum(torch.log(prob) * prob, axis=-1)
    entropy = -torch.sum(m.logits* m.probs, axis=-1)
    return action.to('cpu').numpy(), log_prob, entropy


# multiple rollout
def rollout(env):
    states, rewards, log_probs, entropies = [], [], [], []
    # play an episode
    state = env.reset()
    while True:  
        true_ws = myEnv.new_arrivals(batch_size, thresholds) # generate new arrivals
        extended_state = myEnv.extend_state(state, true_ws, thresholds)
        action, log_prob, entropy = select_action(extended_state) # select an action
        states.append(extended_state)
        log_probs.append(log_prob)
        entropies.append(entropy)
        state, reward, done = env.step(action,true_ws)
#        print(env.state)
#        print(reward)
#        print(env.assignment)
#        print(true_ws)
        rewards.append(reward)
        if done:
            break
    return states, rewards, log_probs, entropies


def train(states, rewards, log_probs, entropies):
    rewards_path, log_probs_paths, avg_reward_path, entropies_path = [], [], [], []
#    for batch in range(len(rewards)):
    R = 0
    P = 0
    for i in reversed(range(len(rewards))):
        R = rewards[i] + args.gamma * R
        rewards_path.insert(0, R)
        P = log_probs[i] + P
        log_probs_paths.insert(0, P)

    log_probs_paths = torch.stack(log_probs_paths).squeeze()
    # rewards_path: np.array(|batch|X|episod|), each element is a reward value
    # log_probs_paths:np.array(|batch|X|episod|), each element is a tensor
    rewards_path = torch.tensor(rewards_path, dtype=torch.float32, device=device).squeeze()
    #entropies_path = torch.stack(entropies)

    states = torch.tensor(states, device=device) 

    value = critic(states).float()  #.view(-1, len_extended_state
   
    # take a backward step for actor  
    entropy_loss = torch.mean(torch.stack(entropies))

    actor_loss = -torch.mean(((rewards_path - value.detach()) * log_probs_paths)  - \
                             args.entropy_coef * entropy_loss
                             )
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    # take a backward step for critic
    loss_fn = torch.nn.MSELoss()
    critic_loss = loss_fn(value, rewards_path)
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()
  
    global actor_weights
    global critic_weights
    actor_total_w = torch.norm(actor.affine1.weight.grad)+ torch.norm(actor.affine2.weight.grad)+torch.norm(actor.affine3.weight.grad)
    actor_weights.append(actor_total_w)
    critic_total_w = torch.norm(critic.affine1.weight.grad)+ torch.norm(critic.affine2.weight.grad)
    critic_weights.append(critic_total_w)
    
    #print('Actor weights:', actor_total_w ,  )
    #print('Critic weights:', critic_total_w,  )
    
    #print(rewards_path)
    result = {}
    result['rew'] = rewards_path.mean().item()
    result['actor_loss'] = actor_loss.item()
    result['critic_loss'] = critic_loss.item()
    result['value'] = torch.mean(value).item()

    return result


   
rws = []
torchMean = []
actor_weights = []
critic_weights = []
inf_counter = 0

def run_training(budget):
    global inf_counter
    for i_episode in range(budget):
        states, rewards, log_probs, entropies = rollout(env)
        if -np.inf in np.array(rewards):
            inf_counter += 1
            print('warning')
        else:
            result = train(states, rewards, log_probs, entropies)
            rws.append(result['rew'])
            torchMean.append(result['value'])
            if i_episode % 20 == 0:
                print(i_episode, result)
            if i_episode % 100 == 0:
                print('actor norm:', torch.norm(torch.cat([i.flatten() for i in actor.parameters()])))


run_training(100000)


plt.close()
temp1 = np.array(actor_weights)
temp2 = temp1[temp1 <=100] 
plt.plot(actor_weights, label = 'actor norm')
plt.plot(critic_weights, label = 'critic norm')
plt.legend()

temp1 = np.array(rws)
temp2 = temp1[temp1 >= -10] 
plt.plot(temp2, 'b-', label = 'rewards')
plt.plot(np.array(torchMean), 'r-' ,label = 'critic value')
plt.xlabel('episodes')
plt.legend()

def simulate_optimal_RL_policy(true_ws, batch_size, num_strata, num_arms, N, thresholds, plot = False):
    new_env = myEnv.trialEnv(state = [],
                         assignment=[],
                         clock = 0,
                         batch_size=1,
                         num_strata=num_strata,
                         num_covs= len(thresholds),
                         num_arms=num_arms,
                         max_time=N,
                         thresholds=thresholds,
                         terminal_signal = False
                         )
    state = new_env.reset()
    counter = 0
    A = []
    while True:  
        extended_state = myEnv.extend_state(state, true_ws[counter], thresholds)
        action, log_prob = select_action(extended_state) # select an action
        assign = [1 if action[0] == ii else 0 for ii in range(num_arms)]
        A.append(assign)
        state, reward, done = new_env.step(action, true_ws[counter])
        counter+=1
        if done:
            break
    r2 = myEnv.find_wd(true_ws, A, plot = plot, figure_name = 'RL_result')
    if abs(r2 + reward[0])>= 0.0001: # reward is negative but r2 is positive
        raise Exception(f'Error: reward is not reported correctly. {r2} != {reward}')
    return new_env, -1 * reward[0]



# run simulation for batchsize = 1 and fixed true_ws
batch_size_sim = 1    
true_ws = []
for i in range(N):
    true_ws.append(myEnv.new_arrivals(batch_size_sim, thresholds)) # generate new arrivals
    
# run simulate_optimal_RL_policy
new_env, reward_RL = simulate_optimal_RL_policy(true_ws, batch_size_sim, num_strata,num_arms, N, thresholds, True)

# run heuristics (extended_pocock)
A_pocock, r_pocock = pocock.main_pocock(true_ws, num_arms, thresholds, N, plot = False)


# exact method (batch arrival)
A_gurobi, r_gurobi = g_a.gurobi_assignment(true_ws, num_arms,  len(thresholds), N, timeLimit= 1000, plot = True, Gplot = False)

# sequential assignment

A_sequential, r_sequential = r_a.sequential_assignment(true_ws,num_arms, N, plot = True)


print(f'rewards: \n actor_critic: \t {reward_RL} \n pocock: \t {r_pocock} \n Gurobi: \t {r_gurobi} \n sequential: \t {r_sequential}')
