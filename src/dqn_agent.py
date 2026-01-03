import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DuelingDQN(nn.Module):
    """Dueling DQN architecture - separates value and advantage streams"""
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for faster learning"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant to avoid zero priority
    
    def __len__(self):
        return self.size

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.0003, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 use_double_dqn=True, use_prioritized_replay=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_replay = use_prioritized_replay
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Use Dueling DQN architecture
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss - more stable than MSE
        
        # Replay memory
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)
            self.beta = 0.4
            self.beta_increment = 0.001
        else:
            self.memory = deque(maxlen=100000)
        
        self.batch_size = 256  # Larger batch for faster learning
        self.min_memory_size = 1000
        
        # Training statistics
        self.training_steps = 0
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        if self.use_prioritized_replay:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        if self.use_prioritized_replay:
            if len(self.memory) < self.min_memory_size:
                return 0
            
            # Sample from prioritized replay buffer
            samples, indices, weights = self.memory.sample(self.batch_size, self.beta)
            if samples is None:
                return 0
            
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            states, actions, rewards, next_states, dones = zip(*samples)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            if len(self.memory) < self.min_memory_size or len(self.memory) < self.batch_size:
                return 0
            
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            weights = torch.ones(self.batch_size).to(self.device)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Next Q values with Double DQN
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use policy net to select action, target net to evaluate
                next_actions = self.policy_net(next_states).argmax(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN
                next_q_values = self.target_net(next_states).max(1)[0]
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute weighted loss
        td_errors = target_q_values - current_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities if using PER
        if self.use_prioritized_replay:
            priorities = td_errors.abs().detach().cpu().numpy()
            self.memory.update_priorities(indices, priorities)
        
        self.training_steps += 1
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def soft_update_target_network(self, tau=0.005):
        """Soft update: target = tau * policy + (1 - tau) * target"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.training_steps = checkpoint.get('training_steps', 0)
        print(f"Model loaded from {filepath}")