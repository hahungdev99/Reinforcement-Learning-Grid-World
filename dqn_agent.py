import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNetwork(nn.Module):
    """
    Deep Q-Network: Neural Network để ước lượng Q-values
    
    Input: State (x, y) được encode thành vector
    Output: Q-values cho tất cả actions
    """
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNetwork, self).__init__()
        
        # Input: 2 features (x, y)
        # Output: 4 Q-values (cho 4 actions)
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)

class ReplayBuffer:
    """
    Experience Replay Buffer
    Lưu trữ experiences và sample ngẫu nhiên để training
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Thêm experience vào buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample một batch experiences ngẫu nhiên"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network Agent
    
    Sử dụng Neural Network thay vì Q-table
    Thêm Experience Replay và Target Network
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001,
                 discount_factor=0.95, epsilon=0.1, epsilon_decay=0.995,
                 epsilon_min=0.01, batch_size=32, buffer_size=10000,
                 hidden_size=64, target_update_freq=100):
        """
        Args:
            state_size: Kích thước của grid (n)
            action_size: Số lượng actions (4)
            learning_rate: Learning rate cho neural network
            discount_factor: Discount factor (γ)
            epsilon: Epsilon cho epsilon-greedy
            epsilon_decay: Tốc độ giảm epsilon
            epsilon_min: Epsilon tối thiểu
            batch_size: Kích thước batch cho training
            buffer_size: Kích thước replay buffer
            hidden_size: Số neurons trong hidden layers
            target_update_freq: Tần suất update target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network chỉ dùng để inference
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training counter
        self.training_steps = 0
    
    def _state_to_tensor(self, state):
        """Chuyển state (x, y) thành tensor"""
        # Normalize state về [0, 1]
        state_array = np.array(state, dtype=np.float32) / self.state_size
        return torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
    
    def choose_action(self, state):
        """
        Chọn action sử dụng epsilon-greedy
        
        Args:
            state: Trạng thái hiện tại (x, y)
        
        Returns:
            action: Hành động được chọn (0-3)
        """
        if np.random.random() < self.epsilon:
            # Exploration
            return np.random.randint(self.action_size)
        else:
            # Exploitation
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def learn(self, state, action, reward, next_state, done):
        """
        Thêm experience vào buffer và train nếu đủ samples
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng
            next_state: Trạng thái tiếp theo
            done: Episode kết thúc chưa
        """
        # Thêm vào replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Train nếu đủ samples
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()
        
        # Giảm epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _train_step(self):
        """Thực hiện một bước training"""
        # Sample batch từ replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Chuyển sang tensors
        states = torch.FloatTensor(states).to(self.device) / self.state_size
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device) / self.state_size
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Tính Q-values hiện tại
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Tính target Q-values sử dụng target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        # Tính loss và backprop
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping để stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def get_q_values(self, state):
        """Lấy Q-values cho một state"""
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]
    
    def get_best_action(self, state):
        """Lấy action tốt nhất (greedy)"""
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def save_model(self, filepath):
        """Lưu model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']