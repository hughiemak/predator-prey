import collections
import numpy as np

class ExperienceBuffer:
	def __init__(self, capacity):
		self.buffer = collections.deque(maxlen=capacity)

	def __len__(self):
		return len(self.buffer)

	def append(self, experience):
		self.buffer.append(experience)
	
	def extend(self, experiences):
		self.buffer.extend(experiences)
	
	def get_random_indices(self, batch_size):
		return np.random.choice(len(self.buffer), batch_size, replace=False)

	def sample(self, batch_size):
		indices = self.get_random_indices(batch_size)
		states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
		return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

	def her_sample(self, batch_size):
		indices = self.get_random_indices(batch_size)
		states, actions, rewards, dones, next_states, goals = zip(*[self.buffer[idx] for idx in indices])
		return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states), np.array(goals)