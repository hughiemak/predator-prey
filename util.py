# Created by Hei Yi Mak 2021 

import numpy as np
import torch
import torch.nn as nn
from pdb import set_trace as debug
from enum import Enum, auto
from collections import Counter

class UncertaintyEstimationMethod(Enum):
	NONE = auto() # Standard state
	RND = auto()

def get_agent_specific_batch(batch, agent_no):
    states, actions, rewards, dones, next_states = batch
    states = get_agent_observations(states, agent_no)
    next_states = get_agent_observations(next_states, agent_no)
    return states, actions, rewards, dones, next_states

def get_tensor_batch(agent_specific_batch, agent_no, device, feed_agent):
    states, actions, rewards, dones, next_states = agent_specific_batch
    states_v = torch.tensor(states).to(device).float()
    next_states_v = torch.tensor(next_states).to(device).float()
    actions_v = torch.tensor(actions).to(device).gather(1,torch.tensor([[agent_no]]*len(actions)).to(device)).squeeze(-1)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    return states_v, actions_v, rewards_v, done_mask, next_states_v #, zeros_v, agent_v

def get_zeros_v(states_v, device):
    zeros_v = torch.zeros(states_v.shape).to(device).float()
    return zeros_v

def get_agent_v(agent_no, batch_size, device, feed_agent):
    agent_v = None 
    if feed_agent:
        agent_v = torch.tensor([[agent_no]]*batch_size).to(device).float()
    return agent_v

def one_hot_embeddings(labels, num_classes):
    return nn.functional.one_hot(labels, num_classes)

def one_hot_embedding(label, num_classes):
    return one_hot_embeddings(torch.tensor([label]), num_classes)

def get_agent_specific_batch_tensor(batch, agent_no, device, feed_agent=False):
    agent_specific_batch = get_agent_specific_batch(batch, agent_no)
    states_v, actions_v, rewards_v, done_mask, next_states_v = get_tensor_batch(agent_specific_batch, agent_no, device, feed_agent)
    zeros_v = get_zeros_v(states_v, device)
    agent_v = get_agent_v(agent_no, len(states_v), device, feed_agent)
    return  states_v, actions_v, rewards_v, done_mask, next_states_v, zeros_v, agent_v

def calc_loss(batch, net, tgt_net, agent_no, gamma, device='cpu', feed_agent=False, predict_action=False):
    states_v, actions_v, rewards_v, done_mask, next_states_v, zeros_v, agent_v = batch
    policy_input = states_v
    tgt_policy_input = next_states_v
    if predict_action:
        policy_input = torch.cat((policy_input, zeros_v), 2)
        tgt_policy_input = torch.cat((tgt_policy_input, zeros_v), 2)
    state_action_values = net(policy_input, agent_nos=agent_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1) # Q(s,a)
    next_state_values = tgt_net(tgt_policy_input, agent_nos=agent_v).max(1)[0] # max_a'(Q(s',a'))
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * gamma + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)

def calc_rnd_loss(batch, rnd_nets, agent, device='cpu'):
    obs_v, actions_v, rewards_v, done_mask, next_obs_v, zeros_v, agent_v = batch
    rnd_pred_nets, rnd_tgt_nets = rnd_nets
    tgt_rnd_embedding = rnd_tgt_nets[agent](obs_v).detach()
    pred_rnd_embedding = rnd_pred_nets[agent](obs_v)
    return nn.MSELoss("sum")(tgt_rnd_embedding, pred_rnd_embedding)/len(obs_v)

def calc_action_prediction_loss(batch, net, agent_no, device='cpu', feed_agent=True):
    states_v, actions_v, rewards_v, done_mask, next_states_v, zeros_v, agent_v = batch
    action_logits = net(torch.cat((states_v, next_states_v), 2), agent_nos=agent_v)
    return nn.CrossEntropyLoss()(action_logits, actions_v)

def get_agent_observations(states, agent_no):
    result = []
    for state in states:
        result.append(get_relative_positions(state, agent_no))
    return np.array(result)

def compute_epsilon(step_idx, epsilon_start, epsilon_final, epsilon_decay_first_step, epsilon_decay_last_step):
	if step_idx < epsilon_decay_first_step:
		return epsilon_start
	else:
		return max(epsilon_final, epsilon_start * (epsilon_decay_last_step - step_idx) / (epsilon_decay_last_step - epsilon_decay_first_step))

def get_relative_positions(s, agent_no):
    # state.shape == (1,d,d) where d is dimension of the environment
    agent_no = agent_no + 1
    state = np.squeeze(np.copy(s))
    agent_coord = tuple([z[0] for z in np.where(state==agent_no)])
    state = np.concatenate((state[agent_coord[0]:], state[:agent_coord[0]]), axis=0)
    state = np.concatenate((state[:,agent_coord[1]:], state[:,:agent_coord[1]]), axis=1)
    for i in range(1,5):
        if i is agent_no:
            state[state==i] = -1
        else:
            state[state==i] = -2
    state[state==5] = -3
    state = state*-1
    state = np.expand_dims(state,axis=0)
    return state

def get_relative_positions_for_prey(state):
    s = get_relative_positions(state, 4)
    s[s==3]=1
    return s

def get_goal(state):
    return get_relative_positions_for_prey(state)

def concat_observations(obs1, obs2):
    return np.concatenate((obs1, obs2), 1)
    
def get_reduced_state(state, agent_no):
    debug()

def get_k_concat_observation_shape(observation_shape, k):
    # obs_shape: (1, n, n)
    z = torch.zeros(observation_shape)
    k_concat_obs_shape = torch.cat((z,)*k, 1).shape
    return k_concat_obs_shape

def get_goal_state(observation_shape):
    goal_state = np.zeros(observation_shape, dtype=np.int16)
    goal_state[0,1,1] = 5
    goal_state[0,0,1] = 1
    goal_state[0,1,2] = 2
    goal_state[0,2,1] = 3
    goal_state[0,1,0] = 4
    return goal_state

def compare_np_arrays(array1, array2):
    return np.array_equal(array1, array2)

def majority_vote(items):
    counter = Counter(items)
    maj = counter.most_common(1)
    if len(maj) > 0:
        return maj[0][0]
    return None

################ HER related functions ################
def get_agent_specific_batch_tensor_her(batch, agent_no, device, feed_agent):
    states, actions, rewards, dones, next_states = get_agent_specific_batch(batch[:-1], agent_no)
    goals = batch[-1]
    state_goals = np.concatenate((states, goals),2)
    next_state_goals = np.concatenate((next_states, goals),2)
    processed_batch = state_goals, actions, rewards, dones, next_state_goals
    zeros_v = get_zeros_v(states, device)
    agent_v = get_agent_v(agent_no, len(states), device, feed_agent)
    tensor_batch = get_tensor_batch(processed_batch, agent_no, device, feed_agent)
    return (*tensor_batch, zeros_v, agent_v)