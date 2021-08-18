import torch
import torch.nn as nn
import numpy as np
import util
from pdb import set_trace as debug

# Convolution Network
class DQN(nn.Module):
	def __init__(self, input_shape, n_actions, predict_action, feed_agent=True):
		super(DQN, self).__init__()

		self.predict_action = predict_action

		self.conv = nn.Sequential(
			nn.Conv2d(input_shape[0], 32, kernel_size=2, stride=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=2, stride=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=1, stride=1),
			nn.ReLU()
		)

		conv_out_size = self._get_conv_out(input_shape)
		if feed_agent:
			conv_out_size = conv_out_size + 1

		self.fc = nn.Sequential(
			nn.Linear(conv_out_size, 512),
			nn.ReLU(),
			nn.Linear(512, n_actions),
		)

	def _get_conv_out(self, shape):
		o = self.conv(torch.zeros(1, *shape))
		return int(np.prod(o.size()))

	def forward(self, x, agent_nos=None):
		conv_out = self.conv(x).view(x.size()[0], -1)
		if agent_nos is not None:
			conv_out = torch.cat((conv_out, agent_nos), dim=-1)
		return self.fc(conv_out)

class LinearDQN(nn.Module):
	def __init__(self, input_shape, n_actions):
		super(LinearDQN, self).__init__()

		self.fc = nn.Sequential(
			nn.Linear(np.prod(input_shape), 256),
			nn.ReLU(),
			nn.Linear(256, n_actions)
		)

	def forward(self, x, agent_nos=None):
		return self.fc(x.view(x.size()[0], -1))

class GreedyAgent:
	def __init__(self, env):
		self.env = env
	
	def __call__(self, agent_nos, epsilon=1/3):
		i = agent_nos
		my_coord = self.env.predator_prey_coords[i]
		prey_coord = self.env.predator_prey_coords[-1]
		best_action = None
		best_dist = None
		if np.random.random() < epsilon:
			best_action_onehot = util.one_hot_embedding(np.random.choice(self.env.nA), self.env.nA)
		else:
			for action in range(self.env.action_space.n):
				next_coord = self.env.get_next_coord(my_coord, action)
				next_coord_y, next_coord_x = next_coord
				prey_coord_y, prey_coord_x = prey_coord
				max_y = max(next_coord_y, prey_coord_y)
				min_y = min(next_coord_y, prey_coord_y)
				dy = min(
					(max_y-min_y), 
					(min_y+self.env.shape[0]-max_y)
				)
				max_x = max(next_coord_x, prey_coord_x)
				min_x = min(next_coord_x, prey_coord_x)
				dx = min(
					(max_x-min_x),
					(min_x+self.env.shape[1]-max_x)
				)
				if best_dist is None or best_dist > dx + dy:
					best_dist = dx + dy
					best_action = action
			best_action_onehot = util.one_hot_embedding(best_action, self.env.nA)
		return best_action_onehot

class RandomNet:
	def __init__(self, env):
		self.env = env

	def __call__(self, policy_input, agent_nos):
		return util.one_hot_embedding(np.random.choice(self.env.nA), self.env.nA)

class RNDNet(nn.Module):
	def __init__(self, input_shape, k=32):
		super(RNDNet, self).__init__()

		self.fc = nn.Sequential(
			nn.Linear(np.prod(input_shape), 256),
			nn.ReLU(),
			nn.Linear(256, k)
		)

	def forward(self, x):
		return self.fc(x.view(x.size()[0], -1))