# Created by Hei Yi Mak 2021 

import collections
import evaluate
import numpy as np
import torch
import torch.optim as optim
import time
import util
from model import DQN, LinearDQN, RNDNet
from pdb import set_trace as debug
from buffer import ExperienceBuffer
from agents import Agents
from experience import Experience, HER_Experience

class Trainer:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        
    def _get_trainer_properties(self):
        observation_shape = self.env.reset().shape
        n_actions = self.env.action_space.n
        n_predators = self.env.n_predators
        device = self.device
        return observation_shape, n_actions, n_predators, device

    def initialize_training(self, lr, rnd_lr, replay_size, centralized, predict_action, feed_agent=True, models_paths=None):
        self.models_paths = models_paths
        observation_shape, n_actions, n_predators, device = self._get_trainer_properties()
        self.buffer = ExperienceBuffer(replay_size)
        self.agents = Agents(self.env, self.buffer, predict_action)

        print(f"predict action: {predict_action}\n-------")
        print(f'feed agent id: {feed_agent}\n-------')

        network_input_shape = observation_shape

        if predict_action:
            network_input_shape = util.get_k_concat_observation_shape(observation_shape, 2)

        if centralized:
            nets, tgt_nets, optimizers = self._init_param_sharing(network_input_shape, n_actions, lr, n_predators, predict_action, feed_agent, device)
            rnd_pred_nets, rnd_tgt_nets, rnd_optimizers = None, None, None # RND does not support parameter sharing
        else:
            nets, tgt_nets, optimizers = self._init_independent_q(network_input_shape, n_actions, lr, n_predators, predict_action, feed_agent, device)
            rnd_pred_nets, rnd_tgt_nets, rnd_optimizers = self._init_rnd_networks(network_input_shape, rnd_lr, n_predators, device)

        print(f'Agent network:\n{nets[0]}\n-------')
        self.nets = nets
        self.tgt_nets = tgt_nets
        self.optimizers = optimizers
        print(f'Agent network:\n{rnd_pred_nets[0]}\n-------')
        self.rnd_pred_nets, self.rnd_tgt_nets, self.rnd_optimizers = rnd_pred_nets, rnd_tgt_nets, rnd_optimizers

    def initialize_her_training(self, lr, replay_size, centralized, predict_action, feed_agent=True):
        observation_shape, n_actions, n_predators, device = self._get_trainer_properties()
        self.her_buffer = ExperienceBuffer(replay_size)

        network_input_shape = util.get_k_concat_observation_shape(observation_shape, 2)

        if centralized:
            nets, tgt_nets, optimizers = self._init_param_sharing(network_input_shape, n_actions, lr, n_predators, predict_action, feed_agent, device)
        else:
            nets, tgt_nets, optimizers = self._init_independent_q(network_input_shape, n_actions, lr, n_predators, predict_action, feed_agent, device)
        
        self.her_nets = nets
        self.her_tgt_nets = tgt_nets
        self.her_optimizers = optimizers
    
    def _init_rnd_networks(self, network_input_shape, lr, n_predators, device):
        rnd_tgt_nets = []
        rnd_pred_nets = []
        optimizers = []
        for i in range(n_predators):
            rnd_target_net = RNDNet(network_input_shape).to(device)
            rnd_predictor_net = RNDNet(network_input_shape).to(device)
            if self.models_paths[i] is not None:
            	print(f"Loading rnd model from path: {self.models_paths[i]}")
            	rnd_target_net.load_state_dict(torch.load(self.models_paths[i] + f"/agent{i}-rnd-tgt-net.dat"))
            	rnd_predictor_net.load_state_dict(torch.load(self.models_paths[i] + f"/agent{i}-rnd-pred-net.dat"))
            rnd_tgt_nets.append(rnd_target_net)
            rnd_pred_nets.append(rnd_predictor_net)
            optimizers.append(optim.RMSprop(rnd_predictor_net.parameters(), lr=lr))
        return rnd_pred_nets, rnd_tgt_nets, optimizers

    def _init_independent_q(self, network_input_shape, n_outputs, lr, n_predators, predict_action, feed_agent, device):
        print("Independent DQNs\n-------")
        nets = []
        tgt_nets = []
        optimizers = []

        for i in range(n_predators):
            net = LinearDQN(network_input_shape, n_outputs).to(device)
            tgt_net = LinearDQN(network_input_shape, n_outputs).to(device)
            # net = DQN(network_input_shape, n_outputs, predict_action, feed_agent=feed_agent).to(device)
            # tgt_net = DQN(network_input_shape, n_outputs, predict_action, feed_agent=feed_agent).to(device)

            if self.models_paths[i] is not None:
            	print(f"Loading model from path: {self.models_paths[i]}")
            	net.load_state_dict(torch.load(self.models_paths[i] + f"/agent{i}-best.dat"))
            	tgt_net.load_state_dict(torch.load(self.models_paths[i] + f"/agent{i}-best.dat"))
            nets.append(net)
            tgt_nets.append(tgt_net)
            optimizers.append(optim.RMSprop(net.parameters(), lr=lr))
        return nets, tgt_nets, optimizers

    def _init_param_sharing(self, network_input_shape, n_outputs, lr, n_predators, predict_action, feed_agent, device):
        print("Parameter Sharing\n-------")
        nets = []
        tgt_nets = []
        optimizers = []

        net = DQN(network_input_shape, n_outputs, predict_action, feed_agent=feed_agent).to(device)
        tgt_net = DQN(network_input_shape, n_outputs, predict_action, feed_agent=feed_agent).to(device)
        optimizer = optim.RMSprop(net.parameters(), lr=lr)
        for _ in range(n_predators):
            nets.append(net)
            tgt_nets.append(tgt_net)
            optimizers.append(optimizer)
        
        return nets, tgt_nets, optimizers

    def initialize_feedback_nets(self, feedbacks):
        feedback_nets = []
        tgt_feedback_nets = []
        optimizers = []
        return feedback_nets, tgt_feedback_nets, optimizers

    def _log_target_task_statistics(self, writer, step_idx, epsilon, median_catch_time, mean_catch_time, catch_time, n_episode):
        writer.add_scalar("epsilon", epsilon, n_episode)
        writer.add_scalar("mean_ep_length_against_episodes", mean_catch_time, n_episode)
    
    def _log_action_predication_statistics(self, writer, step_idx, mean_action_prediction_loss):
        writer.add_scalar("mean_action_prediction_loss", mean_action_prediction_loss, step_idx)

    def _log_evaluation(self, writer, n_episode, mean, median, success_rate):
        writer.add_scalar("eval_mean", mean, n_episode)
        writer.add_scalar("eval_median", median, n_episode)
        writer.add_scalar("eval_success_rate", success_rate, n_episode)
    
    def _log_rnd_statistics(self, writer, n_episode, total_rnd_loss):
        writer.add_scalar("rnd_loss_against_episode", total_rnd_loss, n_episode)

    def _log_cumulative_budget(self, writer, n_episode, budgets, initial_budgets):
        cum_b_ask, cum_b_give = 0, 0
        init_b_ask, init_b_give = initial_budgets
        for b_ask, b_give in zip(*budgets):
            cum_b_ask += init_b_ask - b_ask
            cum_b_give += init_b_give - b_give
        writer.add_scalar("cumulative_b_ask", cum_b_ask, n_episode)
        writer.add_scalar("cumulative_b_give", cum_b_give, n_episode)
        return cum_b_ask, cum_b_give
    
    def _log_uncertainties(self, writer, n_episode, ask_uncertainties_mean, ask_uncertainties_std): #, moving_ask_uncertainties_mean, moving_ask_uncertainties_std):
        writer.add_scalar("ask_uncertainties_mean", ask_uncertainties_mean, n_episode)
        writer.add_scalar("ask_uncertainties_std", ask_uncertainties_std, n_episode)

    def _log_thresholds(self, writer, n_episode, thresholds):
        ask_threshold, give_threshold = thresholds
        writer.add_scalar("ask_threshold", ask_threshold, n_episode)
        writer.add_scalar("give_threshold", give_threshold, n_episode)

    def get_nets_and_optims(self):
        nets = self.nets
        tgt_nets = self.tgt_nets
        optimizers = self.optimizers
        return nets, tgt_nets, optimizers

    def get_her_nets_and_optims(self):
        return self.her_nets, self.her_tgt_nets, self.her_optimizers
    
    def get_rnd_nets_and_optims(self):
        return (self.rnd_pred_nets, self.rnd_tgt_nets), self.rnd_optimizers

    def get_budgets(self, ask, give):
        n_agents = len(self.nets)
        return [ask] * n_agents, [give] * n_agents

    def compute_thresholds(self, thresholds):
        return thresholds
    
    def eval_policies(self, nets, writer, device, step_idx, n_episode):
        mean, median, success_rate = evaluate.eval(100, self.env, nets, device)
        print("steps: %10d, episodes: %6d, eval_mean: %7.3f, eval_median: %3d, success_rate: %7.3f" % (step_idx, n_episode, mean, median, success_rate))
        if writer:
            self._log_evaluation(writer, n_episode, mean, median, success_rate)
        return mean, median, success_rate

    def fill_up_buffer(self, replay_start_size, nets, epsilon, device):
        fill_up_count = replay_start_size
        while fill_up_count > 0:
            fill_up_count -= 1
            _, _ = self.agents.play_step(nets, epsilon, device=device)
        self.agents.reset()

    def train(
        self,
        max_training_episode,
        epsilon_start,
        epsilon_final,
        epsilon_decay_first_step,
        epsilon_decay_last_step,
        device,
        writer,
        save,
        model_dir_path,
        replay_start_size,
        sync_target_steps,
        batch_size,
        gamma):
        # Get nets and optimizers
        nets, tgt_nets, optimizers = self.get_nets_and_optims()
        # Initialize required variables
        epsilon = epsilon_start
        best_eval_mean = None
        step_idx = 0
        n_episode = 0
        # Fill up buffer
        self.fill_up_buffer(replay_start_size, nets, epsilon, device)
        # Training loop
        while n_episode <= max_training_episode:
            epsilon = util.compute_epsilon(step_idx, epsilon_start, epsilon_final, epsilon_decay_first_step, epsilon_decay_last_step)
            _, done_detail = self.agents.play_step(nets, epsilon=epsilon, device=device)
            for agent_no, (net, tgt_net, optimizer) in enumerate(zip(nets, tgt_nets, optimizers)):
                if len(self.buffer) >= replay_start_size:
                    # Optimize NNs for target task
                    batch = self.buffer.sample(batch_size)
                    batch_processed = util.get_agent_specific_batch_tensor(batch, agent_no, device)
                    optimizer.zero_grad()
                    loss_t = util.calc_loss(batch_processed, net, tgt_net, agent_no, gamma, device=device)
                    loss_t.backward()
                    optimizer.step()
            # Sync nets with target nets every C steps
            if step_idx % sync_target_steps == 0:
                for net, tgt_net in zip(nets, tgt_nets):
                    tgt_net.load_state_dict(net.state_dict())
            if done_detail is not None:
                catch_time, _ = done_detail
                print("steps: %10d, episodes: %6d, ep_length: %3d" % (step_idx, n_episode, catch_time))
                if (n_episode % 50 == 0):
                    mean, median, success_rate = self.eval_policies(nets, writer, device, step_idx, n_episode)
                    if (best_eval_mean is None or best_eval_mean > mean) and save is not None:
                        for i, net in enumerate(nets):
                            torch.save(net.state_dict(), model_dir_path + "/" + f"agent{i}-best.dat")
                        if best_eval_mean is not None:
                            print("Best eval mean updated %.3f -> %.3f, model saved" % (best_eval_mean, mean))
                        best_eval_mean = mean
                n_episode += 1
            step_idx += 1
        if (writer):
            writer.close()

    def train_rnd(
        self, 
        max_training_episode, 
        ask_continue_step,
        epsilon_start,
        epsilon_final,
        epsilon_decay_first_step,
        epsilon_decay_last_step,
        # agents,
        device,
        feed_agent,
        writer,
        save,
        model_dir_path,
        env_name,
        replay_start_size,
        sync_target_steps,
        # buffer,
        batch_size,
        gamma,
        predict_action,
        uncertainty_est_method,
        initial_budgets,
        predefined_threshold):

        # Get nets and optimizers
        nets, tgt_nets, optimizers = self.get_nets_and_optims()
        rnd_nets, rnd_optimizers = self.get_rnd_nets_and_optims()
        budgets = self.get_budgets(*initial_budgets) # [b_ask]*4, [b_give]*4

        # Initialize required variables
        epsilon = epsilon_start
        thresholds = predefined_threshold
        ask_uncertainties_means, ask_uncertainties_stds = collections.deque(maxlen=30), collections.deque(maxlen=30)
        best_eval_mean = None
        step_idx = 0
        n_episode = 0
        learn_target_task = True # Toggle this variable to switch between optimizing NN for target task/action prediction

        # fill up buffer
        fill_up_count = replay_start_size
        while fill_up_count > 0:
            fill_up_count -= 1
            _, _ = self.agents.play_step(nets, epsilon, device=device)
        self.agents.reset()
        
        # Training loop
        while n_episode <= max_training_episode:
            step_idx += 1

            epsilon = util.compute_epsilon(step_idx, epsilon_start, epsilon_final, epsilon_decay_first_step, epsilon_decay_last_step)
            total_rnd_loss = None
                
            if uncertainty_est_method is util.UncertaintyEstimationMethod.NONE:
                # play step without uncertainty estimation
                _, done_detail, _, _ = self.agents.play_step_rnd(nets, rnd_nets, budgets, thresholds, enable_advice=False, epsilon=epsilon, device=device)
            elif uncertainty_est_method is util.UncertaintyEstimationMethod.RND:
                # play step with RND uncertainty estimation
                _, done_detail, ask_uncertainties_mean, ask_uncertainties_std = self.agents.play_step_rnd(nets, rnd_nets, budgets, thresholds, enable_advice=True, epsilon=epsilon, device=device)
                if ask_uncertainties_mean is not None and ask_uncertainties_std is not None:
                    if self.agents.t - 1 == 0: 
                        if predefined_threshold is None: # if adaptive threholds are used
                            ask_uncertainties_means.append(ask_uncertainties_mean) # save ask uncertainties mean
                            ask_uncertainties_stds.append(ask_uncertainties_std) # save ask uncertainties std
                            moving_ask_uncertainties_mean = np.mean(ask_uncertainties_means) # compute ask uncertainties moving mean
                            moving_ask_uncertainties_std = np.mean(ask_uncertainties_stds) # compute ask uncertainties movind std
                            thresholds = (moving_ask_uncertainties_mean+moving_ask_uncertainties_std, moving_ask_uncertainties_mean-moving_ask_uncertainties_std) # modify thresholds
                            if writer: # Log the ask and give thresholds.
                                self._log_thresholds(writer, n_episode, thresholds)
                        if writer: # Log the average uncertainty for the initial state of each episode.
                            self._log_uncertainties(writer, n_episode, ask_uncertainties_mean, ask_uncertainties_std) #, moving_ask_uncertainties_mean, moving_ask_uncertainties_std)
            
            if len(self.buffer) >= replay_start_size:

                # Update agent models
                if learn_target_task or not predict_action:
                    if uncertainty_est_method is util.UncertaintyEstimationMethod.RND:
                        total_rnd_loss = 0.0
                    # Optimize NN for target task
                    for agent_no, (net, tgt_net, optimizer) in enumerate(zip(nets, tgt_nets, optimizers)):
                        batch = self.buffer.sample(batch_size)
                        batch_processed = util.get_agent_specific_batch_tensor(batch, agent_no, device, feed_agent)
                        optimizer.zero_grad()
                        loss_t = util.calc_loss(batch_processed, net, tgt_net, agent_no, gamma, device=device, feed_agent=feed_agent, predict_action=predict_action)
                        loss_t.backward()
                        optimizer.step()

                        if uncertainty_est_method is util.UncertaintyEstimationMethod.RND:
                            # Update RND networks
                            rnd_optimizers[agent_no].zero_grad()
                            loss_rnd = util.calc_rnd_loss(batch_processed, rnd_nets, agent_no, device)
                            total_rnd_loss += loss_rnd.item()
                            loss_rnd.backward()
                            rnd_optimizers[agent_no].step()
                    
                else:
                    # Optimize NN for action prediction (learn transition dynamics by predicting a given s and s')
                    if predict_action:
                        losses = []
                        for agent_no, (net, optimizer) in enumerate(zip(nets, optimizers)):
                            batch = self.buffer.sample(batch_size)
                            batch_processed = util.get_agent_specific_batch_tensor(batch, agent_no, device, feed_agent)
                            optimizer.zero_grad()
                            loss_t = util.calc_action_prediction_loss(batch_processed, net, agent_no, device, feed_agent=feed_agent)
                            losses.append(loss_t.item())
                            loss_t.backward()
                            optimizer.step()
                        action_prediction_loss = np.mean(losses) # action prediction loss averaged over all agents
                        action_pred_loss_history.append(action_prediction_loss)
                
                learn_target_task = not learn_target_task

                # Sync nets with target nets every C steps
                if step_idx % sync_target_steps == 0:
                    for net, tgt_net in zip(nets, tgt_nets):
                        tgt_net.load_state_dict(net.state_dict())

            if done_detail is not None:
                catch_time, _ = done_detail
                
                # Log cumulative budget
                cum_b_ask, cum_b_give = 0, 0
                if writer:
                    cum_b_ask, cum_b_give = self._log_cumulative_budget(writer, n_episode, budgets, initial_budgets)
                
                print("steps: %10d, episodes: %6d, ep_length: %3d" % (step_idx, n_episode, catch_time))

                if (n_episode % 50 == 0):
                    mean, median, success_rate = self.eval_policies(nets, writer, device, step_idx, n_episode)
                    if (best_eval_mean is None or best_eval_mean > mean) and save is not None:
                        rnd_pred_nets, rnd_tgt_nets = rnd_nets
                        for i, (net, rnd_pred_net, rnd_tgt_net) in enumerate(zip(nets, rnd_pred_nets, rnd_tgt_nets)):
                            torch.save(net.state_dict(), model_dir_path + "/" + f"agent{i}-best.dat")
                            if uncertainty_est_method is util.UncertaintyEstimationMethod.RND:
                                torch.save(rnd_pred_net.state_dict(), model_dir_path + "/" + f"agent{i}-rnd-pred-net.dat")
                                torch.save(rnd_tgt_net.state_dict(), model_dir_path + "/" + f"agent{i}-rnd-tgt-net.dat")
                        if best_eval_mean is not None:
                            print("Best eval mean updated %.3f -> %.3f, model saved" % (best_eval_mean, mean))
                        best_eval_mean = mean
                
                n_episode += 1

        if (writer):
            writer.close()

    def train_her(
        self,
        max_training_step,
        epsilon_start,
        epsilon_final,
        epsilon_decay_first_step,
        epsilon_decay_last_step,
        device,
        feed_agent,
        success_reward,
        batch_size,
        replay_start_size,
        sync_target_steps,
        gamma,
        writer,
        save,
        model_dir_path,
        env_name):
        # Get nets and optimizers
        nets, tgt_nets, optimizers = self.get_her_nets_and_optims()

        # Initialize required variables
        step_idx = 0
        n_episode = 0
        episode_lens = collections.deque(maxlen=1000)
        best_mean_episode_len = None

        # Set true goal
        observation_shape, _, _, _ = self._get_trainer_properties()
        goal_state = util.get_goal_state(observation_shape)
        true_goal = util.get_goal(goal_state)

        # Define auxiliary functions
        def concat_observations_and_goal(observations, goal):
            return np.array([util.concat_observations(o,goal) for o in observations])

        def concat_observations_and_true_goal(observations):
            return concat_observations_and_goal(observations, true_goal)

        def get_reward(state, goal):
            return int(util.compare_np_arrays(util.get_goal(state), goal))*success_reward*1.0
        
        # Training loop
        while step_idx <= max_training_step:
            step_idx += 1
            epsilon = util.compute_epsilon(step_idx, epsilon_start, epsilon_final, epsilon_decay_first_step, epsilon_decay_last_step)
            detail, done_detail = self.agents.play_step(nets, epsilon, device=device, feed_agent=feed_agent, transform_observations=concat_observations_and_true_goal)

            if done_detail is not None:
                n_episode += 1
                episode_len, episode_transitions = done_detail
                goal = util.get_goal(detail.next_state)
                her_transitions = []

                states, actions, rewards, dones, next_states  = map(np.array, zip(*episode_transitions))
                true_goals = np.array([true_goal]*episode_len)
                goals = np.array([goal]*episode_len)

                for s, a, r, done, s_, tg, g in zip(states, actions, rewards, dones, next_states, true_goals, goals):
                    exp = HER_Experience(s, a, r, done, s_, tg)
                    her_exp = HER_Experience(s, a, get_reward(s_, g), done, s_, g)
                    her_transitions.append(exp)
                    her_transitions.append(her_exp)

                self.her_buffer.extend(her_transitions)

                # Optimize NN for target task
                if len(self.buffer) >= replay_start_size:
                    for i in range(episode_len):
                        for agent_no, (net, tgt_net, optimizer) in enumerate(zip(nets, tgt_nets, optimizers)):
                            batch = self.her_buffer.her_sample(batch_size)
                            batch_processed = util.get_agent_specific_batch_tensor_her(batch, agent_no, device, feed_agent)
                            optimizer.zero_grad()
                            loss_t = util.calc_loss(batch_processed, net, tgt_net, agent_no, gamma, device=device, feed_agent=feed_agent)
                            loss_t.backward()
                            optimizer.step()

                episode_lens.append(episode_len)
                mean_episode_len = np.mean(list(episode_lens)[-100:])
                median_episode_len = np.median(list(episode_lens)[-100:])

                # Log best mean episode duration (target task performance)
                if (len(episode_lens) > 0) and writer:
                    self._log_target_task_statistics(writer, step_idx, epsilon, median_episode_len, mean_episode_len, episode_len, n_episode)

                # Print training progress
                print("steps: %10d, episodes: %6d, ep_length: %3d, median_ep_length: %3d, mean_ep_length: %7.3f, epsilon: %.2f" % (step_idx, n_episode, episode_len, median_episode_len, mean_episode_len, epsilon))

                # Sync nets with target nets every C steps
                if step_idx % sync_target_steps == 0:
                    for net, tgt_net in zip(nets, tgt_nets):
                        tgt_net.load_state_dict(net.state_dict())

                # Save models if best mean episode length > mean episode length
                if (len(episode_lens) > 100) and (best_mean_episode_len is None or best_mean_episode_len > mean_episode_len) and (save is not None):
                    for i, net in enumerate(nets):
                        torch.save(net.state_dict(), model_dir_path + "/" + env_name + f"-agent{i}-best.dat")

                    if best_mean_episode_len is not None:
                        print("Best mean episode length updated %.3f -> %.3f, model saved" % (best_mean_episode_len, mean_episode_len))
                    best_mean_episode_len = mean_episode_len

        if (writer):
            writer.close()