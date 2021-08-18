import gym
from gym import spaces
import numpy as np
from pdb import set_trace as debug
from enum import Enum, auto

class StateType(Enum):
	STANDARD = auto() # Standard state
	STANDARD_ONEHOT = auto()
	RELATIVE = auto() # Relative positions

class PredatorPreyEnv(gym.Env):

	IDLE = 0
	LEFT = 1
	DOWN = 2
	RIGHT = 3
	UP = 4

	MAX_TIMESTEP = 100
	SURROUND_K = 4

	STATE_TYPE = StateType.STANDARD
	RANDOM_PREY_SPAWN = True
	RANDOM_MOVING_PREY = True

	def __init__(self, shape, n_predators=4, success_reward=100):

		self.action_space = spaces.Discrete(5)
		self.nA = self.action_space.n
		self.shape = shape
		self.observation_space = spaces.Box(
			low=0,
			high=n_predators+1,
			shape=self.shape,
			dtype=np.int16)
		self.n_predators = n_predators
		self.predator_prey_coords = [None] * (self.n_predators + 1) # [C1,C2,C3,C4,C5] where C5 is prey's coord
		self.success_reward = success_reward
		self.max_timestep = self.MAX_TIMESTEP

	def reset(self):
		self.current_step = 0
		if self.RANDOM_PREY_SPAWN:
			positions = np.random.choice(self.n_grids(), self.n_predators + 1, replace=False) # Random spawn
		else:
			positions = np.concatenate((np.random.choice(self.n_grids()-1, self.n_predators, replace=False), [self.shape[0]*self.shape[1]-1])) # Simplification: prey always spawns on last grid
		coords = list(map(self.grid_num_to_coord, positions))
		self.predator_prey_coords = coords
		obs = self._get_obs_from_agent_coords(coords)
		return obs

	def step(self, agent_actions):
		assert len(agent_actions)==self.n_predators, "agent_actions should be of length " + str(self.n_predators)

		self.current_step += 1

		if self.RANDOM_MOVING_PREY:
			prey_action = np.random.choice(self.nA, 1)
		else:
			prey_action = [0] # Simplification: prey never moves

		agent_actions = np.concatenate((agent_actions, prey_action))# [A1,A2,A3,A4,A5] where A5 is prey's action

		next_agent_coords = self.predator_prey_coords

		# Process predator actions first, then process prey action
		for agent, action in enumerate(agent_actions):
			curr_coord = self.predator_prey_coords[agent]
			next_coord = self.get_next_coord(curr_coord, action)
			if next_coord not in next_agent_coords:
				next_agent_coords[agent] = next_coord

		self.predator_prey_coords = next_agent_coords

		obs = self._get_obs_from_agent_coords(next_agent_coords)
		
		prey_trapped = self.__prey_is_trapped(next_agent_coords)

		done = self._compute_done(prey_trapped)
		reward = self._compute_reward(prey_trapped)

		info = {}

		return obs, reward, done, info

	def _compute_reward(self, prey_trapped):
		reward = 0
		if (self.current_step <= self.MAX_TIMESTEP and prey_trapped):
			reward = self.success_reward
		return reward
	
	def _compute_done(self, prey_trapped):
		done = False
		if (prey_trapped or self.current_step >= self.MAX_TIMESTEP):
			done = True
		return done

	def get_relative_coords(self, predator_prey_coords):
		predator_coords = predator_prey_coords[:-1]
		prey_coord = predator_prey_coords[-1]
		relative_coords = []
		for coord in predator_coords:
			relative_coords.append(self.get_relative_coord(coord, prey_coord))
		return relative_coords
	
	def get_relative_coord(self, predator_coord, prey_coord):
		predator_y, predator_x = predator_coord
		prey_y, prey_x = prey_coord
		return (predator_y-prey_y)%self.shape[0], (predator_x-prey_x)%self.shape[1]
	
	def surrounded_by_k_predators(self, coords, k):
		prey_coord = coords[-1]
		left = self.get_next_coord(prey_coord, self.LEFT)
		bottom = self.get_next_coord(prey_coord, self.DOWN)
		right = self.get_next_coord(prey_coord, self.RIGHT)
		top = self.get_next_coord(prey_coord, self.UP)
		prey_trapped = False
		scores = 0
		if left in coords:
			scores += 1 
		if bottom in coords:
			scores += 1 
		if right in coords:
			scores += 1
		if top in coords:
			scores += 1
		if scores >= k:
			prey_trapped = True
		return prey_trapped

	def __prey_is_trapped(self, coords):
		prey_trapped = self.surrounded_by_k_predators(coords, self.SURROUND_K)
		return prey_trapped

	def _get_total_distance(self, relative_coords):
		D = 0.0
		for coord in relative_coords:
			dy, dx = coord
			D += min(dy, self.shape[1]-dy)
			D += min(dx, self.shape[0]-dx)
		return D

	def _get_obs_from_agent_coords(self, coords):

		# Ordinary state output
		obs = np.zeros(self.shape, dtype=np.int16)
		# mark agent positions
		for agent, coord in enumerate(coords):
			obs[coord] = agent + 1
		x = np.reshape(obs, (1,-1)).squeeze(axis=0)
		y = np.zeros((x.size, x.max()+1))
		y[np.arange(x.size),x] = 1
		z = y.reshape(obs.shape+(-1,))
		obs_standard = np.expand_dims(obs, axis=0)
		obs_standard_onehot = np.transpose(z,(2,0,1))

		if self.STATE_TYPE == StateType.STANDARD:
			return obs_standard
		else:
			return obs_standard_onehot

	def n_grids(self):
		return self.shape[0] * self.shape[1]

	def grid_num_to_coord(self, grid_num):
		assert grid_num >= 0 and grid_num < self.n_grids(), "grid_num out of bounds"
		return (grid_num // self.shape[1], grid_num % self.shape[1])

	def grid_coord_to_num(self, grid_coord):

		self.verify_coord(grid_coord)

		grid_num = grid_coord[0] * self.shape[1] + grid_coord[1]
		return grid_num

	def get_next_coord(self, curr_coord, action):

		self.verify_coord(curr_coord)

		if (action == self.IDLE):
			return curr_coord
		elif (action == self.LEFT):
			return (curr_coord[0], (curr_coord[1]-1) % self.shape[1])
		elif (action == self.DOWN):
			return ((curr_coord[0]+1) % self.shape[0], curr_coord[1])
		elif (action == self.RIGHT):
			return (curr_coord[0], (curr_coord[1]+1) % self.shape[1])
		elif (action == self.UP):
			return ((curr_coord[0]-1) % self.shape[0], curr_coord[1])
		else:
			assert False, str(action) + " is not a valid action."

	def verify_coord(self, coord):
		assert coord[0] >= 0 and coord[0] < self.shape[0] and coord[1] >= 0 and coord[1] < self.shape[1], str(coord) + "is not a valid coordination."
