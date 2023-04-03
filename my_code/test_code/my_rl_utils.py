from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity) 
        # Create a buffer to store the experience replay data

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated, truncated): 
        self.buffer.append((state, action, reward, next_state, terminated, truncated)) 
        # Save the current state, action, reward, next state, and done value to the buffer

    def sample(self, batch_size: int): 
        '''Randomly sample a batch of data from the buffer'''
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated, truncated = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), terminated, truncated

    def size(self): 
        return len(self.buffer)
        # Return the size of the buffer


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



                