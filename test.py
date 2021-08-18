import torch
import numpy as np
import argparse
from env import PredatorPreyEnv
from pdb import set_trace as debug
from model import DQN, GreedyAgent, RandomNet, LinearDQN
from tqdm import tqdm
from agents import Agents
from buffer import ExperienceBuffer

REPLAY_SIZE = 5000

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--size", help="Set environment size")
	parser.add_argument("--control", action="store_true", help="Enable control")
	return parser.parse_args()
			
args = parse_arguments()
models_path = "models/5x5"
device = "cpu"
size = int(args.size)
env = PredatorPreyEnv(shape=(size, size))
nets = []
observation_shape = env.reset().shape 
n_actions = env.action_space.n
feed_agent = False
control = args.control

for i in range(env.n_predators):
	# model = DQN(observation_shape, n_actions, predict_action=False, feed_agent=feed_agent).to(device)
	model = LinearDQN(observation_shape, n_actions).to(device)
	model.load_state_dict(torch.load(models_path + f"/agent{i}-best.dat", map_location=device))
	# model = None
	nets.append(model)

agents = Agents(env, ExperienceBuffer(REPLAY_SIZE), predict_action=False)

def pretty_print(state):
	state = np.copy(state).astype(str)
	state[state=='5'] = 'X'
	for i in range(4):
		state[state==str(i+1)] = 'O'
	state[state=='0'] = '-'
	print(f'{state}\n')

catch_times = []
rewards = []
iters = 1000
for i in (range(iters) if control else tqdm(range(iters))):
	done_detail = False
	while not done_detail:
		exp, done_detail = agents.play_step(nets, 0., feed_agent=feed_agent)
		if control:
			last_state, joint_action, reward, done, next_state = exp
			pretty_print(last_state)
			input()
		if done_detail:
			catch_time, _ = done_detail
			catch_times.append(catch_time)
			if control:
				pretty_print(next_state)
				print(f'catch time: {catch_time}, reward: {reward}\n')
				input()
			rewards.append(exp.reward)
			
print(np.mean(catch_times))
print(np.median(catch_times))
print((np.sum(rewards)/env.success_reward)/iters)