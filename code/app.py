#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
import time
import matplotlib.pyplot as plt
import streamlit as st


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

###Options
env_name = st.selectbox(
     'Agent type?',
     ('Hopper-v3', 'Humanoid-custom', 'Humanoid-v2'))

st.write('You selected:', env_name)

seed = 1
gamma = st.slider('Gamma: ', 0.0, 1.0, 0.95, 0.01)
alpha = st.slider('Alpha: ', 0.0, 1.0, 0.5, 0.01)
tau = st.slider('Tau: ', 0.0, 1.0, 0.97, 0.01)
l2_reg = st.slider('l2 regularization: ', 0.0, 1.0, 0.9, 0.01)
max_kl = 1e-2
damping = 1e-1
batch_size = st.number_input('batch_size: ', min_value=0, max_value=5000, step=1)
log_interval = st.number_input('log_interval: ', min_value=0, max_value=5000, step=1)
n_epochs = st.number_input('n_epochs: ', min_value=0, max_value=5000, step=1)
render = True

env = gym.make(env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

#env.seed(seed)
torch.manual_seed(seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, max_kl, damping, alpha)

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

#print('observation_space.high: ', env.observation_space.high)
#print('observation_space low: ', env.observation_space.low)
#print('action_space: ', env.action_space)
rewards_plot = []

if st.button('Run training: '):
    plt.ion()
 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = []
    y = []
    line1, = ax.plot(x, y)

            # setting labels
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Updating plot...")

    for i_episode in range(int(n_epochs)):
        memory = Memory()
        #print(i_episode)
        frames = []




        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        while num_steps < batch_size:
            state = env.reset(seed=seed)
            state = running_state(state)

            reward_sum = 0
            for t in range(10000): # Don't infinite loop while learning
                action = select_action(state)
                action = action.data[0].numpy()
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward

                next_state = running_state(next_state)

                mask = 1

                if done:
                    mask = 0

                memory.push(state, np.array([action]), mask, next_state, reward)
                #if i_episode %10 ==0:

                if render and i_episode % log_interval == 0:
                    env.render()
                if done:
                    break

                state = next_state
            num_steps += (t-1)
            num_episodes += 1
            reward_batch += reward_sum

        reward_batch /= num_episodes
        batch = memory.sample()
        update_params(batch)
        #print(frames)
      #  save_frames_as_gif(frames)

        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum, reward_batch))

                #frames.append(env.render())
            if i_episode > 0:
                fig, ax = plt.subplots()
                plt.plot(rewards_plot)
                st.pyplot(fig)
        rewards_plot.append(reward_batch)

