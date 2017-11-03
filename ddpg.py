
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *
from copy import deepcopy
import torch.nn.functional as F

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


criterion = nn.MSELoss()


class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):

        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions

        actor_net_cfg = {
            'hidden1':32,
            'hidden2':32,
            'hidden3':32,
            'init_w':args.init_w
        }

        critic_net_cfg = {
            'hidden1':64,
            'hidden2':64,
            'hidden3':64,
            'init_w':args.init_w
        }

        self.actor = Actor(self.nb_states, self.nb_actions, **actor_net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **actor_net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **critic_net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **critic_net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(
            size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True
        self.best_reward = -10


    def update_policy(self, shared_model, args):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size, shared=args.use_more_states, num_states=args.num_states)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic_optim.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        if args.shared:
            ensure_shared_grads(self.critic, shared_model.critic)

        self.critic_optim.step()

        # Actor update
        self.actor_optim.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        if args.shared:
            ensure_shared_grads(self.actor, shared_model.actor)
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def share_memory(self):
        self.critic.share_memory()
        self.actor.share_memory()

    def add_optim(self, actor_optim, critic_optim):
        self.actor_optim  = actor_optim
        self.critic_optim  = critic_optim

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def update_models(self, agent):
        self.actor = deepcopy(agent.actor)
        self.actor_target = deepcopy(agent.actor_target)
        self.critic = deepcopy(agent.critic)
        self.critic_target = deepcopy(agent.critic_target)
        self.actor_optim  = deepcopy(agent.actor_optim)
        self.critic_optim  = deepcopy(agent.critic_optim)

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        self.a_t = action
        return action

    def train(self):
        self.critic.train()
        self.actor.train()

    def state_dict(self):
        return [self.actor.state_dict(),
                self.actor_target.state_dict(),
                self.critic.state_dict(),
                self.critic_target.state_dict()]

    def load_state_dict(self, list_of_dicts):
        self.actor.load_state_dict(list_of_dicts[0])
        self.actor_target.load_state_dict(list_of_dicts[1])
        self.critic.load_state_dict(list_of_dicts[2])
        self.critic_target.load_state_dict(list_of_dicts[3])

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)