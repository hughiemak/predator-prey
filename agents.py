import torch
import torch.nn as nn
import numpy as np
import util
from pdb import set_trace as debug
from enum import Enum
from experience import Experience, Feedback_Experience
from model import GreedyAgent

class AgentConfigItem(Enum):
	FEEDBACK = "feedback"

class Agents:
	def __init__(self, env, exp_buffer, predict_action):
		self.env = env
		self.exp_buffer = exp_buffer
		self.reset()
		self.config = {}
		self.predict_action = predict_action
		self.greedy_policy = GreedyAgent(self.env)
		
	def use_feedback(self, feedbacks, feedback_buffer):
		self.config[AgentConfigItem.FEEDBACK] = feedbacks
		self.feedback_buffer = feedback_buffer

	def reset(self):
		self.state = self.env.reset()
		self.t = 0
		return self.state

	def _state_to_agent_obs(self, agent, state):
		return util.get_agent_observations([np.copy(state)], agent)

	def _get_action_from_policy(self, policy, state, agent, feed_agent=False, device="cpu", transform_observations=None):
		# agent: agent id
		# feed_agent: if True, feed agent id to policy
		observation = self._state_to_agent_obs(agent, state)
		if transform_observations:
			assert self.predict_action is False, "HER with action prediction is not supported yet."
			observation = transform_observations(observation)
		state_v = torch.tensor(observation).to(device).float()
		zeros_v = torch.zeros(state_v.shape).to(device).float()
		agent_v = None
		if feed_agent:
			agent_v = torch.tensor([[agent]]).to(device).float()
		policy_input = state_v
		if self.predict_action:
			policy_input = torch.cat((policy_input, zeros_v), 2)
		
		# q_vals_v = self.greedy_policy(agent)
		q_vals_v = policy(policy_input, agent_nos=agent_v)
		action = torch.argmax(q_vals_v, dim=1).item()
		return action

	def get_joint_action(self, nets, epsilon, feed_agent, device, transform_observations=None):
		joint_action = []
		# Individual exploration
		# for i, net in enumerate(nets):
		# 	if np.random.random() < epsilon:
		# 		action = self.env.action_space.sample()
		# 	else:
		# 		action = self._get_action_from_policy(net, self.state, i, feed_agent=feed_agent, device=device)
		# 	joint_action.append(action)

		# Group exploration
		x = np.random.random()
		for i, net in enumerate(nets):
			if x < epsilon:
				action = self.env.action_space.sample()
			else:
				action = self._get_action_from_policy(net, self.state, i, feed_agent=feed_agent, device=device, transform_observations=transform_observations)
			joint_action.append(action)

		return joint_action

	def get_episode_experience(self):
		t = self.t
		episode_exp = list(self.exp_buffer.buffer)[-t:]
		return episode_exp

	def epsilon_greedy_exploration(self, epsilon, nets, agent, current_obs, device):
		x = np.random.random()
		if x < epsilon:
			action = self.env.action_space.sample()
		else:
			current_obs_v = torch.tensor(current_obs).to(device).float()
			q_vals_v = nets[agent](current_obs_v)
			action = torch.argmax(q_vals_v, dim=1).item()
		return action
	
	def compute_uncertainty(self, agent, obs, rnd_nets, device):
		rnd_pred_nets, rnd_tgt_nets = rnd_nets
		obs_v = torch.tensor(obs).to(device).float()
		tgt_rnd_embedding = rnd_tgt_nets[agent](obs_v)
		pred_rnd_embedding = rnd_pred_nets[agent](obs_v)
		return torch.nn.MSELoss(reduction="sum")(tgt_rnd_embedding, pred_rnd_embedding) # ||G(o)-G_hat(o)||**2
	
	def ask_for_advice(self, agent, obs, nets, rnd_nets, b_gives, thres_give, device):
		action = None
		advices = []

		for other_agent in range(len(nets)):
			if other_agent is not agent:
				if b_gives[other_agent] > 0:
					uncertainty = self.compute_uncertainty(other_agent, obs, rnd_nets, device)
					u_give = uncertainty.item()
					if u_give < thres_give:
						b_gives[other_agent] -= 1
						# Early Advising
						obs_v = torch.tensor(obs).to(device).float()
						q_vals_v = nets[other_agent](obs_v)
						advice = torch.argmax(q_vals_v, dim=1).item()
						advices.append(advice)
		action = util.majority_vote(advices)
		return action

	def play_step_rnd(self, nets, rnd_nets, budgets, thresholds, enable_advice, epsilon=0.0, device="cpu"):
		assert len(nets) == self.env.n_predators, "Expect " + str(self.env.n_predators) + " agents but " + str(len(nets)) + " agents are provided"
		assert len(nets) == len(rnd_nets[0]), "Expect " + str(len(nets)) + " rnd_nets but " + str(len(rnd_nets[0])) + " rnd_nets are provided"
		self.t += 1
		b_asks, b_gives = budgets
		joint_action = []
		ask_uncertainties = []
		current_observations = []
		for agent in range(len(nets)):
			current_obs = self._state_to_agent_obs(agent, self.state)
			current_observations.append(current_obs)
			uncertainty = self.compute_uncertainty(agent, current_obs, rnd_nets, device)
			ask_uncertainties.append(uncertainty.item())

		for agent in range(len(nets)):
			action = None
			current_obs = current_observations[agent]

			if enable_advice:
				if thresholds is not None:
					thres_ask, thres_give = thresholds
					if b_asks[agent] > 0:
						if ask_uncertainties[agent] > thres_ask:
							# ask for advice
							action = self.ask_for_advice(agent, current_obs, nets, rnd_nets, b_gives, thres_give, device)
			if action is None:
				action = self.epsilon_greedy_exploration(epsilon, nets, agent, current_obs, device)
			else:
				b_asks[agent] -= 1
			joint_action.append(action)
		last_state = self.state
		next_state, reward_E, is_done, _ = self.env.step(joint_action)
		reward = reward_E
		exp = Experience(last_state, joint_action, reward, is_done, next_state)	
		self.exp_buffer.append(exp)
		self.state = next_state

		done_detail = None
		ask_uncertainties_mean = np.mean(ask_uncertainties)
		ask_uncertainties_std = np.std(ask_uncertainties)

		if is_done:
			episode_transitions = self.get_episode_experience()
			catch_time = self.t
			done_detail = (catch_time, episode_transitions)
			self.reset()
			
		return exp, done_detail, ask_uncertainties_mean, ask_uncertainties_std

	def play_step(self, nets, epsilon=0.0, device="cpu", feed_agent=False, transform_observations=None):
		assert len(nets) == self.env.n_predators, "Expect " + str(self.env.n_predators) + " agents but " + str(len(nets)) + " agents are provided"
		self.t += 1
		joint_action = []
		for agent in range(len(nets)):
			observation = self._state_to_agent_obs(agent, self.state)
			action = self.epsilon_greedy_exploration(epsilon, nets, agent, observation, device)
			joint_action.append(action)

		last_state = self.state
		next_state, reward, is_done, _ = self.env.step(joint_action)

		exp = Experience(last_state, joint_action, reward, is_done, next_state)	
		self.exp_buffer.append(exp)
		self.state = next_state

		done_detail = None

		if is_done:
			episode_transitions = self.get_episode_experience()
			catch_time = self.t
			done_detail = (catch_time, episode_transitions)
			self.reset()

		return exp, done_detail