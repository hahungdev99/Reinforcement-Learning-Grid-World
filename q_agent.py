import numpy as np

class QLearningAgent:
    """
    Q-Learning Agent sử dụng Q-table
    
    Có thể dễ dàng kế thừa class này để tạo DQN Agent
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=0.1, epsilon_decay=0.995, 
                 epsilon_min=0.01):
        """
        Args:
            state_size: Kích thước của grid (n)
            action_size: Số lượng actions (4: up, down, left, right)
            learning_rate: Tốc độ học (α)
            discount_factor: Hệ số chiết khấu (γ)
            epsilon: Xác suất khám phá
            epsilon_decay: Tốc độ giảm epsilon
            epsilon_min: Giá trị epsilon tối thiểu
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Khởi tạo Q-table: [state_space_size, action_size]
        # State space size = n * n (vì có n*n positions)
        self.q_table = np.zeros((state_size * state_size, action_size))
    
    def _state_to_index(self, state):
        """Chuyển state (x, y) thành index"""
        x, y = state
        return x * self.state_size + y
    
    def choose_action(self, state):
        """
        Chọn action sử dụng epsilon-greedy policy
        
        Args:
            state: Trạng thái hiện tại (x, y)
        
        Returns:
            action: Hành động được chọn (0-3)
        """
        # Epsilon-greedy: khám phá vs khai thác
        if np.random.random() < self.epsilon:
            # Exploration: chọn action ngẫu nhiên
            return np.random.randint(self.action_size)
        else:
            # Exploitation: chọn action tốt nhất từ Q-table
            state_idx = self._state_to_index(state)
            return np.argmax(self.q_table[state_idx])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Cập nhật Q-table sử dụng Q-learning update rule
        
        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái tiếp theo
            done: Episode đã kết thúc chưa
        """
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)
        
        # Lấy Q-value hiện tại
        current_q = self.q_table[state_idx, action]
        
        # Tính Q-value mục tiêu
        if done:
            # Nếu episode kết thúc, không có future reward
            target_q = reward
        else:
            # Q-learning: lấy max Q-value của state tiếp theo
            max_next_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.discount_factor * max_next_q
        
        # Cập nhật Q-value
        self.q_table[state_idx, action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Giảm epsilon (giảm khám phá theo thời gian)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_q_values(self, state):
        """Lấy tất cả Q-values cho một state"""
        state_idx = self._state_to_index(state)
        return self.q_table[state_idx]
    
    def get_best_action(self, state):
        """Lấy action tốt nhất (không có exploration)"""
        state_idx = self._state_to_index(state)
        return np.argmax(self.q_table[state_idx])
    
    def save_q_table(self, filename):
        """Lưu Q-table ra file"""
        np.save(filename, self.q_table)
    
    def load_q_table(self, filename):
        """Load Q-table từ file"""
        self.q_table = np.load(filename)