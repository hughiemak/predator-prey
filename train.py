import os
import util
import torch
import argparse
import numpy as np
import torch.nn as nn
import warnings
import collections
import util
from method import TrainingMethod
from env import PredatorPreyEnv
from pdb import set_trace as debug
from tqdm import tqdm
from buffer import ExperienceBuffer
from tensorboardX import SummaryWriter
from agents import Agents
from trainer import Trainer

os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings("ignore", category=UserWarning)

RESTORE_MODEL_PATHS = [None, None, None, None] # models/path-to-model-folder
FEED_AGENT = False
CENTRALIZED = False # Parameter sharing
TRAINING_METHOD = TrainingMethod.STANDARD
UNCERTAINTY_EST_METHOD = util.UncertaintyEstimationMethod.NONE # Method
B_ASK, B_GIVE = 0, 0 # 1000000, 1000000 # None, None # None: Unlimited budget
THRES_ASK_THRES_GIVE = None # (1.0, 0.5) # None: Use (moving mean +/- moving std) as ask and give thresholds

ENV_NAME = "predator-prey-env"
ENV_DEFAULT_SIZE = 4
SUCCESS_REWARD = 100
FEEDBACK_CHOICES = [-0.01, 0.0, 0.01]
ENABLE_FEEDBACK = False
PREDICT_ACTION = False

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
RND_LEARNING_RATE = 1e-7
SYNC_TARGET_STEPS = 500
REPLAY_START_SIZE = 5000

EPSILON_DECAY_FISRT_STEP = 0
EPSILON_DECAY_LAST_STEP = 100000
EPSILON_START = 0.05
EPSILON_FINAL = 0.05

ASK_CONTINUE_STEP = None
MAX_TRAINING_STEP = 100000
MAX_TRAINING_EPISODE = 1000

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", action="store_true", help="Enable cuda")
	parser.add_argument("--size", default=ENV_DEFAULT_SIZE, help="Size of the environment")
	parser.add_argument("--writer", default=None, help="Directory name for training log")
	parser.add_argument("--save", default=None, help="Name of directory to save the models")
	return parser.parse_args()

def setup(args):
	# set up environment
	env = PredatorPreyEnv(shape=(int(args.size), int(args.size)), success_reward=SUCCESS_REWARD)
	device = torch.device("cuda" if args.cuda else "cpu")
	# set up device
	print(f'-------\ncuda: {args.cuda}\nenvironment size: {args.size}\nwriter: {args.writer}\ndirectory: {args.save}\n-------')
	# set up model directory path
	model_dir_path = None
	if args.save is not None:
		model_dir_path = "models/" + args.save
		os.mkdir(model_dir_path)
	# set up writer
	writer = None
	if (args.writer):
		writer = SummaryWriter(log_dir=f"runs/{args.writer}")
	# set up save
	save = args.save
	return env, model_dir_path, device, writer, save


if __name__=="__main__":
	args = parse_arguments()
	env, model_dir_path, device, writer, save = setup(args)

	# Initialize networks and optimizers
	trainer = Trainer(env, device)
	trainer.initialize_training(lr=LEARNING_RATE, rnd_lr=RND_LEARNING_RATE, replay_size=REPLAY_SIZE, centralized=CENTRALIZED, predict_action=PREDICT_ACTION, feed_agent=FEED_AGENT, models_paths=RESTORE_MODEL_PATHS)

	if ENABLE_FEEDBACK:
		feedback_buffer = ExperienceBuffer(REPLAY_SIZE)
		agents.use_feedback(FEEDBACK_CHOICES, feedback_buffer)

	# Initialize required variables
	epsilon = EPSILON_START
	catch_times = collections.deque(maxlen=1000)
	best_mean_catch_time = None
	has_saved_model = False
	step_idx = 0
	n_episode = 0

	if TRAINING_METHOD is TrainingMethod.STANDARD:
		trainer.train(
			MAX_TRAINING_EPISODE,
			EPSILON_START,
			EPSILON_FINAL,
			EPSILON_DECAY_FISRT_STEP,
			EPSILON_DECAY_LAST_STEP,
			device,
			writer,
			save,
			model_dir_path,
			REPLAY_START_SIZE,
			SYNC_TARGET_STEPS,
			BATCH_SIZE,
			GAMMA)
	else:
		trainer.train_rnd(
			MAX_TRAINING_EPISODE,
			ASK_CONTINUE_STEP,
			EPSILON_START,
			EPSILON_FINAL,
			EPSILON_DECAY_FISRT_STEP,
			EPSILON_DECAY_LAST_STEP,
			device,
			FEED_AGENT,
			writer,
			save,
			model_dir_path,
			ENV_NAME,
			REPLAY_START_SIZE,
			SYNC_TARGET_STEPS,
			BATCH_SIZE,
			GAMMA,
			PREDICT_ACTION,
			UNCERTAINTY_EST_METHOD,
			(B_ASK, B_GIVE),
			THRES_ASK_THRES_GIVE
			)