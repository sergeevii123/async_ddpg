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
parser.add_argument('--rmsize', default=100000, type=int, help='memory size')
parser.add_argument('--init_w', default=0.003, type=float, help='')
parser.add_argument('--window_length', default=1, type=int, help='')
parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
parser.add_argument('--discount', default=0.9, type=float, help='')
parser.add_argument('--epsilon', default=10000, type=int, help='linear decay of exploration policy')
parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
parser.add_argument('--load_weights', dest="load_weights", action='store_true', help='load weights for actor and critic')
parser.add_argument('--shared', dest="shared", action='store_true')
parser.add_argument('--update_train_agents', default=3, type=int)
parser.add_argument('--use_more_states', dest="use_more_states", action='store_true')
parser.add_argument('--num_states', default=4, type=int)


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def test_new_state_dict(agent, current_reward, env, num_tests=3, use_more_states=False, num_states=4):
    episode_reward = 0
    for i in range(num_tests):
        observation = deepcopy(env.reset())
        if use_more_states:
            observations = deque(list(observation for i in range(2**num_states)), 2**num_states)
            # observations = deque(list(observation for i in range(num_states)), num_states)
        agent.reset(observation)
        done = False

        while not done:
            if use_more_states:
                cur_observations = list()
                for i in range(num_states):
                    cur_observations.append(list(observations)[2**i-1])
                action = agent.select_action(np.concatenate(list(cur_observations)).ravel().tolist())
                # action = agent.select_action(np.concatenate(list(observations)).ravel().tolist())
            else :
                action = agent.select_action(observation)

            observation2, reward, done, info = env.step(action)
            observation = deepcopy(observation2)
            if use_more_states and observation:
                observations.appendleft(observation)
            episode_reward += reward

    episode_reward /= num_tests
    prRed("current reward {:.3f} from new model {:.3f}".format(current_reward, episode_reward))
    if current_reward < episode_reward:
        return True
    else:
        return False

def train(rank, args, ns, best_result, actor_optim, critic_optim, shared_model, debug=True):
    torch.manual_seed(args.seed + rank)

    env = RunEnv(False)
    env.seed(args.seed + rank)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    if args.use_more_states:
        agent = DDPG(nb_states*args.num_states, nb_actions, args)
    else :
        agent = DDPG(nb_states, nb_actions, args)

    if args.shared:
        agent.add_optim(actor_optim, critic_optim)

    agent.train()
    if args.load_weights:
        agent.load_weights("weights")
    agent.is_training = True
    step = episode = episode_steps = 0
    observation = None
    done = True
    episode_reward = 0.
    observations = None
    last_reward = -10
    while True:
        if args.shared:
            agent.load_state_dict(shared_model.state_dict())

        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)
            if args.use_more_states:
                observations = deque(list(observation for i in range(2**args.num_states)), 2**args.num_states)
                # observations = deque(list(observation for i in range(args.num_states)), args.num_states)

        if step <= args.warmup:
            action = agent.random_action()
        elif args.use_more_states:
            cur_observations = list()
            for i in range(args.num_states):
                cur_observations.append(list(observations)[2 ** i - 1])
            action = agent.select_action(np.concatenate(list(cur_observations)).ravel().tolist())
        else :
            action = agent.select_action(observation)

        observation2, reward, done, info = env.step(action)
        observation = deepcopy(observation2)
        if observation and args.use_more_states:
            observations.appendleft(observation)

        if args.use_more_states:
            cur_observations = list()
            for i in range(args.num_states):
                cur_observations.append(list(observations)[2 ** i - 1])
            agent.observe(reward, np.concatenate(list(cur_observations)).ravel().tolist(), done)
        else :
            agent.observe(reward, observation, done)

        if step > args.warmup :
            for i in range(5):
                agent.update_policy(shared_model, args)

        step += 1
        episode_steps += 1
        episode_reward += reward

        if done:
            if args.use_more_states:
                cur_observations = list()
                for i in range(args.num_states):
                    cur_observations.append(list(observations)[2 ** i - 1])
                agent.memory.append(
                    np.concatenate(list(cur_observations)).ravel().tolist(),
                    agent.select_action(np.concatenate(list(cur_observations)).ravel().tolist()),
                    0., False
                )
            else :
                agent.memory.append(
                    observation,
                    agent.select_action(observation),
                    0., False
                )

            if step > args.warmup and best_result.value < episode_reward and episode_reward > last_reward:
                best_model = ns.best_model
                best_model.load_state_dict(agent.state_dict())
                if debug: prLightPurple("best reward: {:.3f} current reward: {:.3f} updated best model from agent {}"
                                        .format(best_model.best_reward, episode_reward, rank))
                best_model.best_reward = episode_reward
                agent.best_reward = episode_reward
                ns.best_model = best_model
                best_result.value = episode_reward
                last_reward = best_result.value
            elif step > args.warmup and episode % 10 == 0 and episode > 0 and args.update_train_agents > 0 \
                    and best_result.value > episode_reward and best_result.value > agent.best_reward:
                best_model = ns.best_model
                test_agent = deepcopy(agent)
                test_agent.load_state_dict(best_model.state_dict())
                if test_new_state_dict(test_agent, episode_reward, env, args.update_train_agents, use_more_states=args.use_more_states, num_states=args.num_states):
                    agent = test_agent
                    agent.best_reward = best_model.best_reward
                    if debug: prGreen("best result {:.3f} updated agent {}".format(best_model.best_reward, rank))
                    last_reward = best_model.best_reward


            observation = None
            observations = None
            if debug: prCyan('agent_{:02d} ep:{} ep_steps:{} reward:{:.3f} '.format(rank, episode, episode_steps, episode_reward ))
            episode_steps = 0
            episode += 1
            episode_reward = 0.




def test(rank, args, ns, best_result):

    env = RunEnv(False)
    env.seed(args.seed + rank)
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    if args.use_more_states:
        agent = DDPG(nb_states*args.num_states, nb_actions, args)
    else :
        agent = DDPG(nb_states, nb_actions, args)

    if args.load_weights:
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
            if args.use_more_states:
                observations = deque(list(observation for i in range(2**args.num_states)), 2**args.num_states)
                # observations = deque(list(observation for i in range(args.num_states)), args.num_states)

        if best_result.value > best_episode_reward and step > args.warmup:
            best_model = deepcopy(ns.best_model)
            test_agent = deepcopy(agent)
            test_agent.load_state_dict(best_model.state_dict())
            if test_new_state_dict(test_agent, episode_reward, env, use_more_states=args.use_more_states, num_states=args.num_states) :
                agent = test_agent
                agent.best_reward = best_model.best_reward
                prRed("updated test agent from ns {:.3f}".format(best_model.best_reward))
            last_reward = best_result.value
            observation = None

        episode_steps = 0
        episode_reward = 0.

        # start episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)
            if args.use_more_states:
                observations = deque(list(observation for i in range(2**args.num_states)), 2**args.num_states)
                # observations = deque(list(observation for i in range(args.num_states)), args.num_states)

        done = False
        while not done:
            if args.use_more_states:
                cur_observations = list()
                for i in range(args.num_states):
                    cur_observations.append(list(observations)[2 ** i - 1])
                action = policy(np.concatenate(list(cur_observations)).ravel().tolist())
            else :
                action = policy(observation)

            observation, reward, done, info = env.step(action)
            if args.use_more_states and observation:
                observations.appendleft(observation)

            episode_reward += reward
            episode_steps += 1
            step+=1

        if episode % 50 ==0 and episode != 0:
            print("saving models")
            os.makedirs("weights", exist_ok=True)
            agent.save_model("weights")
        episode+=1
        observation = None
        observations = None
        current_best_result = best_result.value
        best_episode_reward = max(episode_reward, best_episode_reward)
        best_result.value = max(episode_reward, current_best_result - 0.05)
        print('#Ep{}: episode_reward:{:.3f} episode_steps:{} br: {:.3f} -> {:.3f}'.format(
            episode,episode_reward, episode_steps, current_best_result, best_result.value))




if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')
    args = parser.parse_args()

    # uncomment when it's fixed in pytorch
    # torch.manual_seed(args.seed)

    env = RunEnv(False)
    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    mgr = Manager()
    ns = mgr.Namespace()
    if args.use_more_states:
        ns.best_model = DDPG(nb_states*args.num_states, nb_actions, args)
        shared_model = DDPG(nb_states*args.num_states, nb_actions, args)
    else :
        ns.best_model = DDPG(nb_states, nb_actions, args)
        shared_model = DDPG(nb_states, nb_actions, args)

    if args.load_weights:
        shared_model.load_weights("weights")
        ns.best_model.load_weights("weights")

    actor_optim = critic_optim = None

    if args.shared:
        actor_optim = shared_adam.SharedAdam(shared_model.actor.parameters(), lr=args.rate)
        critic_optim = shared_adam.SharedAdam(shared_model.actor.parameters(), lr=args.rate)
        shared_model.add_optim(actor_optim, critic_optim)

        actor_optim.share_memory()
        critic_optim.share_memory()
        shared_model.share_memory()

    processes = []
    best_result = mp.Value('f', -10)

    p = mp.Process(target=test, args=(args.num_processes, args, ns, best_result))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, ns, best_result, actor_optim, critic_optim, shared_model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
