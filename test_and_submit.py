from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp
from multiprocessing import Manager
import numpy as np
from copy import deepcopy
import shared_adam
from ddpg import DDPG
from random import random
from osim.env import *
from util import *
from collections import deque
# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='async_ddpg')

parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4, help='how many training processes to use (default: 4)')
parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
parser.add_argument('--hidden1', default=64, type=int, help='hidden num of first fully connect layer')
parser.add_argument('--hidden2', default=64, type=int, help='hidden num of second fully connect layer')
parser.add_argument('--hidden3', default=64, type=int, help='hidden num of second fully connect layer')
parser.add_argument('--init_w', default=0.05, type=float, help='')
parser.add_argument('--window_length', default=1, type=int, help='')
parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
parser.add_argument('--bsize', default=200, type=int, help='minibatch size')
parser.add_argument('--discount', default=0.99, type=float, help='')
parser.add_argument('--epsilon', default=500000, type=int, help='linear decay of exploration policy')
parser.add_argument('--warmup', default=200, type=int, help='time without training but only filling the replay memory')
parser.add_argument('--load', default=1000, type=int, help='time without training but only filling the replay memory')
parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
parser.add_argument('--load_weights', dest="load_weights", action='store_true', help='load weights for actor and critic')
parser.add_argument('--shared', dest="shared", action='store_true')
parser.add_argument('--update_train_agents', dest="update_train_agents", action='store_true',
                    help='copy best weights from train agents to each other')


def test(rank, args):

    env = RunEnv(True)
    env.seed(args.seed + rank)
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states*4, nb_actions, args)
    agent.load_weights("weights")
    agent.is_training = False
    agent.eval()

    done = True
    policy = lambda x: agent.select_action(x, decay_epsilon=False)
    last_reward = -10
    episode = 0
    observation = None
    observations = None
    episode_reward = 0.
    step = 0
    best_episode_reward = -10
    while True:
        # reset at the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)
            observations = deque([observation, observation, observation, observation], 4)

        episode_steps = 0
        episode_reward = 0.

        done = False
        while not done:

            action = policy(np.concatenate(list(observations)).ravel().tolist())

            observation, reward, done, info = env.step(action)
            if observation:
                observations.appendleft(observation)

            episode_reward += reward
            episode_steps += 1
            step+=1

        episode+=1
        observation = None
        observations = None
        best_episode_reward = max(episode_reward, best_episode_reward)
        print('#Ep{}: episode_reward:{:.3f} episode_steps:{} '.format(episode,episode_reward, episode_steps))




if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')
    args = parser.parse_args()

    test(11, args)
