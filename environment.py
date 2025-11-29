import numpy as np

class GridEnvironment:
    """
    Môi trường lưới nxn cho bài toán tìm đường
    - Start: (0, 0)
    - Goal: (n-1, n-1)
    - Actions: 0=Up, 1=Down, 2=Left, 3=Right
    """
    
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.current_pos = self.start_pos
        
        # Action mapping: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = 4
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
    
    def reset(self):
        """Reset về vị trí bắt đầu"""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action):
        """
        Thực hiện một hành động
        
        Returns:
            next_state: Trạng thái tiếp theo
            reward: Phần thưởng
            done: Có hoàn thành episode không
        """
        # Tính toán vị trí mới
        delta = self.actions[action]
        new_pos = (
            self.current_pos[0] + delta[0],
            self.current_pos[1] + delta[1]
        )
        
        # Kiểm tra xem có đi ra ngoài biên không
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos
            
            # Kiểm tra có đến đích không
            if self.current_pos == self.goal_pos:
                return self.current_pos, 100.0, True  # Đến đích: +100
            else:
                return self.current_pos, -1.0, False  # Bước đi bình thường: -1
        else:
            # Đi ra ngoài biên: phạt -10 và giữ nguyên vị trí
            return self.current_pos, -10.0, False
    
    def _is_valid_position(self, pos):
        """Kiểm tra vị trí có hợp lệ không"""
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def get_state_space_size(self):
        """Trả về số lượng states có thể có"""
        return self.grid_size * self.grid_size
    
    def state_to_index(self, state):
        """Chuyển state (x, y) thành index cho Q-table"""
        x, y = state
        return x * self.grid_size + y
    
    def index_to_state(self, index):
        """Chuyển index thành state (x, y)"""
        x = index // self.grid_size
        y = index % self.grid_size
        return (x, y)