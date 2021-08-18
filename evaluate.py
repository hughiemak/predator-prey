import numpy as np
import util
import torch
from env import PredatorPreyEnv
from model import DQN, GreedyAgent
from pdb import set_trace as debug
from tqdm import tqdm

def get_net_output(net, obs_batch, device):
    obs_ten = torch.tensor(obs_batch).to(device).float()
    q_vals = net(obs_ten)
    actions_i = torch.argmax(q_vals, dim=-1)
    actions_i = actions_i.cpu().numpy()
    return actions_i

def eval(n, env, nets, device):
    with torch.no_grad():
        shape = env.shape
        envs = [PredatorPreyEnv(shape=shape) for _  in range(n)]
        states = []
        for env in envs:
            state = env.reset()
            states.append(state)

        results = [None] * len(envs)

        for t in range(env.max_timestep):
            episode_steps = t+1
            observations = []
            for i, _ in enumerate(nets):
                observations.append(util.get_agent_observations(states, i))
            actions = []
            for i, net in enumerate(nets):
                actions_i = get_net_output(net, observations[i], device)
                actions.append(actions_i)
            actions = np.array(actions)
            for j, env in enumerate(envs):
                joint_a = actions[:,j].tolist()
                state, reward, done, info = env.step(joint_a)
                states[j] = state
                if done and results[j] is None:
                    results[j] = (episode_steps, reward)
        ep_lens, rewards = zip(*results)
        mean = np.mean(ep_lens)
        median = np.median(ep_lens)
        success_rate = (np.sum(rewards) / env.success_reward) / n
        return mean, median, success_rate
