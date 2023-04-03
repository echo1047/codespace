import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from gymnasium.utils.save_video import save_video
from tqdm import tqdm

import echo.my_code.my_rl_utils as my_rl_utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(torch.nn.Module):
    '''
    The actor takes a state as input and outputs a probability distribution over actions.
    '''
    def __init__(self, state_size, hidden_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
    
class ActorContinuous(torch.nn.Module):
    def __init__(self, state_size, hidden_size, action_size, action_bound):
        super(ActorContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc_mu = torch.nn.Linear(hidden_size, action_size)
        self.fc_std = torch.nn.Linear(hidden_size, action_size)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob
    
    
class Critic(torch.nn.Module): #? 连续状态网络和离散状态网络似乎没有多大的区别(还是有的)
    '''
    The critic takes a state and an action as input and outputs a scalar value that estimates the expected return.
    '''
    def __init__(self, state_size, hidden_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_out = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, s, a):
        cat = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SAC:
    def __init__(self, state_size, hidden_size, action_size, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device):
        self.actor = Actor(state_size, hidden_size, action_size).to(device)
        
        # training network: choose the best action
        self.critic_1 = Critic(state_size, hidden_size, action_size).to(device) 
        self.critic_2 = Critic(state_size, hidden_size, action_size).to(device)
        
        # target network: calculate the target value
        self.target_critic_1 = Critic(state_size, hidden_size, action_size).to(device)
        self.target_critic_2 = Critic(state_size, hidden_size, action_size).to(device)
        
        self.target_critic_1.load_state_dict(self.critic_1.state_dict()) 
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True, device=device) # log(0.01)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.target_entropy = target_entropy

    def take_action(self, state): 
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)[0]
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def calculate_td_target(self, rewards, next_states, terminateds, truncateds):
        '''
        SAC stands for Soft Actor-Critic, which is a reinforcement learning algorithm for continuous action spaces. It uses two types of networks: target networks and training networks.

        Target networks are used to compute the target values for the Q-function and the policy updates. They are obtained by copying the parameters of the training networks periodically or by using a moving average of the training network parameters. Target networks help to stabilize the learning process by reducing the correlation between the target and the current values.

        Training networks are used to learn the Q-function and the policy parameters from the data collected by interacting with the environment. They are updated by minimizing a loss function that depends on the target values computed by the target networks. Training networks help to improve the performance of the agent by learning from its experience.
        '''
        next_action_probs = self.actor(next_states)
        _, next_actions = torch.max(next_action_probs, dim=1)
        next_actions = next_actions.view(-1, 1)
        # 加log是为了计算log概率，而不是概率。log概率有一些优点，例如可以避免数值下溢，可以简化梯度计算，可以增加数值稳定性等。
        # 加log是为了使用REINFORCE算法或其他基于策略梯度的强化学习算法。这些算法的核心思想是利用策略的log概率和回报的乘积来估计策略性能的梯度，并用梯度上升来优化策略。
        # 加log可以使得梯度的形式更加简洁和直观。加log是为了使用交叉熵损失函数或其他基于最大似然的损失函数。这些损失函数的目标是最大化数据的对数似然，即最小化数据和模型输出之间的交叉熵。加log可以使得损失函数的形式更加简洁和直观。
        log_next_action_probs = torch.log(next_action_probs + 1e-8)
        # 计算熵, H(p) = -sum(p(x)log(p(x)))
        entropys = -torch.sum(next_action_probs * log_next_action_probs, dim=1, keepdim=True)
        next_q1_values = self.target_critic_1(next_states, next_action_probs)
        next_q2_values = self.target_critic_2(next_states, next_action_probs)
        min_q_values = torch.min(next_q1_values, next_q2_values)
        td_targets = rewards + self.gamma * (min_q_values - self.log_alpha.exp() * entropys) * (1 - terminateds) * (1 - truncateds)
        return td_targets
    
    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # 标量则将行向量转换为列向量
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        terminateds = torch.tensor(transition_dict['terminateds'], dtype=torch.float).view(-1, 1).to(self.device)
        truncateds = torch.tensor(transition_dict['truncateds'], dtype=torch.float).view(-1, 1).to(self.device)
        # update two critic networks. Using td_target to update critic networks.        
        td_targets = self.calculate_td_target(rewards, next_states, terminateds, truncateds)
        critic_q_values_1 = self.critic_1(states, action_p)
        critic_q_values_2 = self.critic_2(states, actions)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_targets.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_targets.detach()))
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        
        # update actor network. Using q_value to update actor network.
        action_probs = self.actor(states)
        log_action_probs = torch.log(action_probs + 1e-8)
        entropys = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)
        q_values_1 = self.critic_1(states, action_probs)
        q_values_2 = self.critic_2(states, action_probs)
        min_q_values = torch.min(q_values_1, q_values_2, dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropys - min_q_values) # hands-on-RL书中的损失
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update alpha parameter
        alpha_loss = torch.mean(self.log_alpha.exp() * (entropys - self.target_entropy).detach())
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step() #todo 明天跑起来
        
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2) 


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    '''
    在新的API中，done被分成两部分：terminated表示环境是否终止（例如因为任务完成、失败等），truncated表示环境是否截断（例如因为时间限制或者不属于任务MDP的原因）。这样做是为了消除done信号的歧义，让学习算法能够更好地区分不同的结束原因。
    '''
    return_list = []
    for i in range(10): # 要显示的进度条数量
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar: # 一个进度条
            for i_episode in range(int(num_episodes/10)): # for1 一幕
                episode_return = 0
                state, _ = env.reset() # 获取环境初始状态
                terminated, truncated = False, False
                while not (terminated or truncated): # for2
                    action = agent.take_action(state) # 根据策略选择动作
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, terminated, truncated) # 存入回放池
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size: # for3
                        buffer_states, buffer_actions, buffer_rewards, buffer_next_states, buffer_terminateds, buffer_truncateds = replay_buffer.sample(batch_size) # 采样N个元组
                        transition_dict = {'states': buffer_states, 
                                           'actions': buffer_actions, 
                                           'next_states': buffer_next_states, 
                                           'rewards': buffer_rewards, 
                                           'terminateds': buffer_terminateds, 
                                           'truncateds': buffer_truncateds}
                        agent.update(transition_dict) # 更新
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


if __name__ == "__main__":
    actor_lr = 1e-3
    critic_lr = 1e-2
    alpha_lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    target_entropy = -1
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v1' # Atari状态连续,动作离散(且一维)的环境
    env = gym.make(env_name)
    replay_buffer = my_rl_utils.ReplayBuffer(buffer_size)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = SAC(state_size, hidden_dim, action_size, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)

    train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
    # return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)