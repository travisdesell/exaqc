from collections import deque
import torch
import random
import numpy as np

class UDReplayBuffer:
    """
    Uncertainty-Driven Replay Buffer to store (state, action, reward, next state, done, uncertainty)
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self._capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self._memory = deque(maxlen=capacity)
        self._priorities = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, u_ep, u_al, unc_ratio):
        transition = (state, action, reward, next_state, done, u_ep, u_al)
        self._memory.append(transition)
        self._priorities.append(unc_ratio)
    
    def sample(self, batch_size):
        total = self.__len__()
        
        self.increase_beta()
        
        priorities = np.array(self._priorities)

        # Convert priorities to probabilities (proportional sampling)
        probabilities = priorities / np.sum(priorities)

        idxs = np.random.choice(total, batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = (total * probabilities) ** -self.beta
        weights /= np.max(weights)  # Normalize for stability
        states, actions, rewards, next_states, dones, u_eps, u_als = zip(*[self._memory[idx] for idx in idxs])
        weights = [weights[idx] for idx in idxs]
        
        states = torch.stack(states, dim=0)
        next_states = torch.stack(next_states, dim=0)
        
        return (states, actions, rewards, next_states, dones, u_eps, u_als, idxs, weights)

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities.abs()):
            self._priorities[idx] = (abs(priority.item()) + self.epsilon) ** self.alpha

    def increase_beta(self):
        self.beta = np.min([1.0, self.beta + self.beta_increment])

    def __len__(self):
        return len(self._memory)
