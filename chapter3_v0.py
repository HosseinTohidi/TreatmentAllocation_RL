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
import Heuristics.kk as kk
import Gurobi_assignment as g_a
import random_assignment as r_a
import CA_RO 




device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu', 0) #args.gpu_num)
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)') #0.01
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument("--use_gpu", type=bool, default=True)
parser.add_argument("--gpu_num", type=int, default=3)
parser.add_argument("--batchSize", type=int, default=2, help='number of episodes at each training step')
parser.add_argument('--actor_lr',  type=float, default=0.00001, help='entropy term coefficient (default: 0.01)')
parser.add_argument('--critic_lr', type=float, default=0.00001, help='entropy term coefficient (default: 0.01)') #1 not 5 before
parser.add_argument('--method', type=str, default='fc', help='fc | att')
args, unknown = parser.parse_known_args()

CA_RO_uncertainty_set = True

batch_size = 10 #args.batchSize 10 before , changed to 20 for case of 5 arms
num_arms = 3
N = 60
#thresholds = [[0,1],
#              [5,10,15,20,40],
#              [10,20,30],
#              [-5,-3,-1,0,1,3,5]] 
thresholds = [[0,1],
              [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
              [0,0.25,0.5,0.75, 1],
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
        self.affine1 = nn.Linear(len_extended_state, 512) #256 before
        self.affine2 = nn.Linear(512, 512)
        self.affine3 = nn.Linear(512, num_arms)
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
    m = Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    #entropy = - torch.sum(torch.log(probs) * probs, axis=-1)
    entropy = -torch.sum(m.logits* m.probs, axis=-1)
    return action.to('cpu').numpy(), log_prob, entropy

# multiple rollout
def rollout(env):
    states, rewards, log_probs, entropies = [], [], [], []
    # play an episode
    state = env.reset()
    True_WS = []
    counter = 0
    while True:  
        if CA_RO_uncertainty_set:
            #true_ws = myEnv.new_arrivals2(batch_size, thresholds, N, True_WS) # generate new arrivals
            #print(true_ws)
            #True_WS.append(true_ws)
            if len(True_WS) ==0:
                True_WS = myEnv.new_arrivals3(batch_size, thresholds, N) # generate new arrivals
            #print(true_ws)
            true_ws = True_WS[counter]
            counter += 1
        else:
            true_ws = myEnv.new_arrivals(batch_size, thresholds) # generate new arrivals
        #print(true_ws)
        extended_state = myEnv.extend_state(state, true_ws, thresholds)
        action, log_prob, entropy = select_action(extended_state) # select an action
        states.append(extended_state)
        log_probs.append(log_prob)
        entropies.append(entropy)
        state, reward, done = env.step(action,true_ws)
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
                             ) #
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

save_model_check = False
load_model_check = False

def save_model(state, filename = 'model_weight50_2objs_200k_RO_set_n20.pth.tar'):
    print('=> Saving the model')
    torch.save(state, filename)




def load_model(check_point):
    print('=> Loading the model')
    actor.load_state_dict(check_point['actor_dict'])
    critic.load_state_dict(check_point['critic_dict'])
    actor_optim.load_state_dict(check_point['actor_optim'])
    critic_optim.load_state_dict(check_point['critic_optim'])
    
if save_model_check:
    check_point = {'actor_dict': actor.state_dict(),
               'critic_dict': critic.state_dict(),
               'actor_optim' : actor_optim.state_dict(),
               'critic_optim' : critic_optim.state_dict()}
    save_model(check_point)
if load_model_check:
    load_model(torch.load("model_weight50_2objs_130k_2arms.pth.tar"))

    

run_training(250000)
    



plt.close()
temp1 = np.array(actor_weights)
temp2 = temp1[temp1 <=100] 
plt.plot(temp1, label = 'actor norm')
plt.plot(critic_weights, label = 'critic norm')
plt.legend()

temp = np.array(rws)
temp3 = temp[::100]
temp2 = temp[temp>-5]
plt.plot(-1*temp[:], 'b-')
#plt.plot(np.array(torchMean), 'r-' ,label = 'critic value')
plt.xlabel('iterations')
plt.xticks(np.arange(0,100000+25000,25000))
plt.ylabel('reward')
plt.savefig('.//figs//final figs//2arms_RO_set_20 patients//reward.PNG', dpi = 200)

dll = pd.DataFrame(rws)
dll.to_csv('5arms_rws.csv')
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
        action, log_prob,_ = select_action(extended_state) # select an action
        assign = [1 if action[0] == ii else 0 for ii in range(num_arms)]
        A.append(assign)
        state, reward, done = new_env.step(action, true_ws[counter])
        counter+=1
        if done:
            break
    r2 = myEnv.find_wd(true_ws, A, plot = plot, figure_name = 'RL_result')
    #if abs(r2 + reward[0])>= 0.0001: # reward is negative but r2 is positive
    #    raise Exception(f'Error: reward is not reported correctly. {r2} != {reward}')
    return new_env, A, r2

n_weight = 50
m_weight = 1
def comp_imbalance(A,num_arms):
    #np.array(A).sum(axis = 0).max() - np.array(A).sum(axis = 0).min()
    return abs(np.array(A).sum(axis = 0) - len(A)/num_arms).sum()

def create_sample_test_all(N, thresholds,num_arms,num_strata,Gurobi_timeLimit = 1000, plot = True, num_sample = 1 ):
    sol = []
    for sample in range(num_sample):    
        # run simulation for batchsize = 1 and fixed true_ws
        batch_size_sim = 1    
        #true_ws = []
        True_WS = []
        
        if CA_RO_uncertainty_set:
            True_WS = myEnv.new_arrivals3(batch_size_sim, thresholds, N) # generate new arrivals
            #print(true_ws)
            #True_WS.append(true_ws_tmp)
        else:
            for i in range(N):
                true_ws_tmp = myEnv.new_arrivals(batch_size_sim, thresholds) # generate new arrivals
                True_WS.append(true_ws_tmp)

        true_ws = copy.deepcopy(True_WS)
        
        #for i in range(N):
        #    true_ws.append(myEnv.new_arrivals(batch_size_sim, thresholds)) # generate new arrivals           
       
        # run simulate_optimal_RL_policy
        new_env, A_RL, reward_RL = simulate_optimal_RL_policy(true_ws, batch_size_sim, num_strata,num_arms, N, thresholds, plot= plot)
        #imballance_RL = np.array([new_env.state[batch].sum(axis =0).max() - new_env.state[batch].sum(axis =0).min() for batch in range(new_env.batch_size)])/new_env.num_covs
        #imballance_RL = imballance_RL[0]
        imballance_RL = comp_imbalance(A_RL, num_arms)
        WD_RL = (reward_RL - m_weight* imballance_RL)/n_weight 
       
        # run heuristics (extended_pocock)
        A_pocock, r_pocock = pocock.main_pocock(true_ws, num_arms, thresholds, N, plot = plot)
        imballance_pocock =  comp_imbalance(A_pocock, num_arms)
        WD_pocock = (r_pocock - m_weight* imballance_pocock)/n_weight
        
        # run heuristics (extended_pocock)
        try:
            A_kk, r_kk = kk.kk_assignment(true_ws, num_arms, N, plot = plot)
            imballance_kk = comp_imbalance(A_kk,num_arms)
            WD_kk = (r_kk - m_weight*imballance_kk)/n_weight
        except:
            WD_kk = 0
            imballance_kk = 0
       
        # exact method (batch arrival)
        #A_gurobi, r_gurobi = g_a.gurobi_assignment(true_ws, num_arms,  len(thresholds), N, timeLimit= 1000, plot = plot, Gplot = False)
        r_gurobi= 0
        
        # sequential assignment
        A_sequential, r_sequential = r_a.sequential_assignment(true_ws,num_arms, N, plot = plot)
        imballance_sequential = comp_imbalance(A_sequential,num_arms)
        WD_sequential = (r_sequential - m_weight*imballance_sequential)/n_weight

        
        # completely random
        A_random, r_random = r_a.random_assignment(true_ws,num_arms, N, plot = plot)
        imballance_random = comp_imbalance(A_random,num_arms)
        WD_random = (r_random - m_weight*imballance_random)/n_weight


        # CA_RO
        A_RO, r_RO = CA_RO.CA_RO(true_ws, num_arms, N, plot = plot)
        imballance_RO = comp_imbalance(A_RO,num_arms)
        WD_RO = (r_RO - m_weight*imballance_RO)/n_weight


        sol.append([WD_RL, WD_pocock, WD_kk, r_gurobi, WD_sequential, WD_random, WD_RO, imballance_RL, imballance_pocock, imballance_kk,imballance_random, imballance_RO])
        
    df = pd.DataFrame(data = sol, columns = ['RL', 'pocock', 'kk', 'gurobi', 'sequential', 'random','CA_RO','imballance_RL', 'imballance_pocock', 'imballance_kk','imballance_random', "imballance_CA_RO"])
    return df
df = create_sample_test_all(N, thresholds,num_arms,num_strata,Gurobi_timeLimit = 100, plot = False, num_sample = 100 )

df['new_RL']     = n_weight * df['RL'] + m_weight*df['imballance_RL']
df['new_CA_RO']  = n_weight * df['CA_RO'] + m_weight*df['imballance_CA_RO']
df['new_pocock'] = n_weight * df['pocock'] + m_weight*df['imballance_pocock']
df['new_random'] = n_weight * df['random'] + m_weight*df['imballance_random']
df['new_kk']     = n_weight * df['kk'] + m_weight*df['imballance_kk']

#df.to_csv('final_csv_150k.csv')


dg7 = df['new_RL'] <= df['new_kk']
print(dg7[dg7 == True].shape)

dg7 = df['new_RL'] <= df['new_CA_RO']
print(dg7[dg7 == True].shape)

dg7 = df['new_RL'] <= df['new_pocock']
print(dg7[dg7 == True].shape)

dg7 = df['new_RL'] <= df['new_random']
print(dg7[dg7 == True].shape)


dg7 = df['imballance_RL'] <= df['imballance_kk']
print(dg7[dg7 == True].shape)

dg7 = df['imballance_RL'] <= df['imballance_CA_RO']
print(dg7[dg7 == True].shape)

dg7 = df['imballance_RL'] <= df['imballance_pocock']
print(dg7[dg7 == True].shape)

dg7 = df['imballance_RL'] <= df['imballance_random']
print(dg7[dg7 == True].shape)


dg2 = df['RL'] <= df['kk']
print(dg2[dg2 == True].shape)

dg3 = df['RL'] <= df['CA_RO']
print(dg3[dg3 == True].shape)

dg4 = df['RL'] <= df['pocock']
print(dg4[dg4 == True].shape)

dg5 = df['RL'] <= df['random']
print(dg5[dg5 == True].shape)

df['RL'].head()
chart = sns.boxplot(data= df[['imballance_RL','imballance_CA_RO']])
chart = sns.boxplot(data= df[['RL','CA_RO']])


chart = sns.boxplot(data= df[['new_RL','new_CA_RO']])
plt.xticks(np.arange(2), ['RL','CA_RO'])
plt.ylabel('Total reward')
plt.savefig('.//figs//final figs//2arms_RO_set_20 patients//2arms_new_only_caro_and_RL_total_reward.PNG')

chart = sns.boxplot(data= df[['new_RL','new_CA_RO','new_pocock', 'new_random','new_kk']]) #
plt.xticks(np.arange(5), ['RL','CA_RO','Pocock', 'Random','KK'])  #
plt.ylabel('Total reward')
plt.savefig('.//figs//final figs//2arms_RO_set_20 patients//2arms_new_total_reward.PNG')


chart = sns.boxplot(data= df[['RL','CA_RO','pocock', 'random','kk']]) #
plt.xticks(np.arange(5), ['RL','CA_RO','Pocock', 'Random','KK']) #
plt.ylabel('Total Wasserstein Distance')
plt.savefig('.//figs//final figs//2arms_RO_set_20 patients//2arms_wd.PNG')

chart = sns.boxplot(data= df[['imballance_RL', 'imballance_CA_RO','imballance_pocock', 'imballance_random', 'imballance_kk']]) #
plt.xticks(np.arange(5), ['RL','CA_RO','Pocock', 'Random', 'KK']) #
plt.ylabel('Total Imbalance')
plt.savefig('.//figs//final figs//2arms_RO_set_20 patients//2arms_imbalance.PNG')



plt.style.use('seaborn-white')
fig, ax1  = plt.subplots(figsize=(7.8, 5.51))
props = dict(widths=0.7,patch_artist=True, medianprops=dict(color="gold"))
box1=ax1.boxplot(df['RL'].values, positions=[0], **props)
ax2 = ax1.twinx()
box2=ax2.boxplot(df['imballance_RL'].values,positions=[1], **props)
ax1.set_xticklabels(['WD-RL', 'imballance'])
plt.ylabel = 'RL'
for b in box1["boxes"]+box2["boxes"]:
    b.set_facecolor(next(ax1._get_lines.prop_cycler)["color"])
plt.show()

plt.style.use('seaborn-white')
fig, ax1  = plt.subplots(figsize=(7.8, 5.51))
props = dict(widths=0.7,patch_artist=True, medianprops=dict(color="gold"))
box1=ax1.boxplot(df['CA_RO'].values, positions=[0], **props)
ax2 = ax1.twinx()
box2=ax2.boxplot(df['imballance_CA_RO'].values,positions=[1], **props)
ax1.set_xticklabels(['WD', 'imballance'])
for b in box1["boxes"]+box2["boxes"]:
    b.set_facecolor(next(ax1._get_lines.prop_cycler)["color"])
plt.show()


chart = sns.boxplot(data= df[['new_RL','new_CA_RO']])


df = pd.read_csv('AllModelsIncludingRO_twoObjs.csv')
