import torch
import numpy as np
import argparse
from env import PredatorPreyEnv
from pdb import set_trace as debug
from model import DQN, GreedyAgent, RandomNet, LinearDQN
from tqdm import tqdm
from agents import Agents
from buffer import ExperienceBuffer
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--size", help="Set environment size")
	parser.add_argument("--model", help="Set model path")
	return parser.parse_args()

args = parse_arguments()
models_path = args.model
device = "cpu"
size = int(args.size)
env = PredatorPreyEnv(shape=(size, size))
nets = []
observation_shape = env.reset().shape 
n_actions = env.action_space.n
replay_size = 5000
feed_agent = False

for i in range(env.n_predators):
	model = LinearDQN(observation_shape, n_actions).to(device)
	model.load_state_dict(torch.load(models_path + f"/agent{i}-best.dat", map_location=device))
	nets.append(model)

agents = Agents(env, ExperienceBuffer(replay_size), predict_action=False)

def visualize():
	agents.reset()
	done_detail = None
	states = []
	catch_time = None
	while not done_detail:
		exp, done_detail = agents.play_step(nets, 0., feed_agent=feed_agent)
		last_state, joint_action, reward, done, next_state = exp
		states.append(last_state)
		# if control:
		# 	last_state, joint_action, reward, done, next_state = exp
		# 	pretty_print(last_state)
		# 	input()
		if done_detail:
			catch_time, _ = done_detail
			# catch_times.append(catch_time)
			states.append(next_state)
			# if control:
			# 	pretty_print(next_state)
			# 	print(f'catch time: {catch_time}, reward: {reward}\n')
			# 	input()
			# rewards.append(exp.reward)

	def animate(states, n_predators):
		def processed_state(state):
			state = np.copy(state)
			state = state.squeeze()
			for i in range(n_predators):
			    state[state==i+1]=1
			state[state==5]=2
			return state

		Writer = animation.writers['ffmpeg']
		writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)

		a = np.random.rand(10,10)

		fig, ax = plt.subplots()
		gridline_color = 'lightgrey'
		my_cmap = matplotlib.colors.ListedColormap(['w', 'b', 'r'])
		ax.set_xticks(np.arange(-0.5, 5, 1))
		ax.set_yticks(np.arange(-0.5, 5, 1))
		ax.grid(linestyle='-', linewidth=1, c=gridline_color)
		plt.xticks(color='w')
		plt.yticks(color='w')
		ax.tick_params(tick1On=False)
		plt.setp(ax.spines.values(), color=gridline_color)

		def animate(i):
		    im = ax.imshow(processed_state(states[i]), cmap=my_cmap)

		ani = animation.FuncAnimation(fig,animate, frames=len(states), interval=200, blit=False, repeat=False)
		# ani.save('im.gif', writer='imagemagick', fps=2)
		plt.show()

	animate(states, env.n_predators)
	if catch_time:
		print(f"Catch Time: {catch_time}")

if __name__=="__main__":
	visualize()
