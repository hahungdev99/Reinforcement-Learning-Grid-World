import streamlit as st
import numpy as np
import time
from environment import GridEnvironment
from dqn_agent import DQNAgent
from visualization import visualize_grid, visualize_training_stats, visualize_q_values

st.set_page_config(page_title="DQN Grid Navigation", layout="wide")

st.title("🧠 Deep Q-Network (DQN): Grid Navigation")
st.markdown("""
Using **Neural Network** instead of Q-table to learn pathfinding
- 🔥 Experience Replay: Learn from past experiences
- 🎯 Target Network: Stable training
- 🚀 Scales to larger state spaces
""")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

# Grid size
grid_size = st.sidebar.slider("Grid size (n × n)", 3, 15, 5)

# Training parameters
st.sidebar.subheader("Training Parameters")
episodes = st.sidebar.number_input("Episodes", 100, 10000, 2000, 100)
learning_rate = st.sidebar.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
discount_factor = st.sidebar.slider("Discount factor (γ)", 0.01, 1.0, 0.95, 0.01)
epsilon = st.sidebar.slider("Epsilon (exploration)", 0.0, 1.0, 1.0, 0.01)
epsilon_decay = st.sidebar.slider("Epsilon decay", 0.9, 0.999, 0.995, 0.001)
batch_size = st.sidebar.slider("Batch size", 16, 128, 32, 16)
hidden_size = st.sidebar.slider("Hidden layer size", 32, 256, 64, 32)

# Initialize session state
if 'trained_dqn' not in st.session_state:
    st.session_state.trained_dqn = False
    st.session_state.agent_dqn = None
    st.session_state.env_dqn = None
    st.session_state.training_history_dqn = None

# Create environment and agent
env = GridEnvironment(grid_size)

# Training section
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎓 Train DQN Agent")
    
    if st.button("🚀 Start DQN Training", type="primary"):
        # Tạo agent mới
        agent = DQNAgent(
            state_size=grid_size,
            action_size=4,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            hidden_size=hidden_size
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Training metrics
        rewards_history = []
        steps_history = []
        epsilon_history = []
        
        # Training loop
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < grid_size * grid_size * 2:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            rewards_history.append(total_reward)
            steps_history.append(steps)
            epsilon_history.append(agent.epsilon)
            
            # Update progress
            if episode % 10 == 0:
                progress_bar.progress((episode + 1) / episodes)
                avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                avg_steps = np.mean(steps_history[-100:]) if len(steps_history) >= 100 else np.mean(steps_history)
                status_text.text(f"Episode {episode+1}/{episodes} | Avg Reward: {avg_reward:.2f} | Avg Steps: {avg_steps:.1f} | ε: {agent.epsilon:.3f} | Buffer: {len(agent.replay_buffer)}")
        
        progress_bar.progress(1.0)
        status_text.success(f"✅ Training completed: {episodes} episodes!")
        
        # Save to session state
        st.session_state.trained_dqn = True
        st.session_state.agent_dqn = agent
        st.session_state.env_dqn = env
        st.session_state.training_history_dqn = {
            'rewards': rewards_history,
            'steps': steps_history,
            'epsilon': epsilon_history
        }

with col2:
    st.header("📊 Info")
    if st.session_state.trained_dqn:
        st.metric("Status", "✅ Trained")
        history = st.session_state.training_history_dqn
        st.metric("Avg Reward (last 100 episodes)", 
                  f"{np.mean(history['rewards'][-100:]):.2f}")
        st.metric("Avg Steps (last 100 episodes)", 
                  f"{np.mean(history['steps'][-100:]):.1f}")
        st.metric("Replay Buffer size", 
                  f"{len(st.session_state.agent_dqn.replay_buffer)}")
    else:
        st.metric("Status", "⏳ Not trained")

# Training statistics
if st.session_state.trained_dqn:
    st.header("📈 Training Statistics")
    visualize_training_stats(st.session_state.training_history_dqn)

# Testing section
st.header("🎮 Test Agent")

if st.session_state.trained_dqn:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Run 1 Episode", type="secondary"):
            agent = st.session_state.agent_dqn
            env = st.session_state.env_dqn
            
            state = env.reset()
            path = [state]
            total_reward = 0
            steps = 0
            done = False
            
            st.write("### 🗺️ Path:")
            step_container = st.empty()
            grid_container = st.empty()
            
            # Greedy mode
            old_epsilon = agent.epsilon
            agent.epsilon = 0
            
            while not done and steps < grid_size * grid_size * 2:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                
                state = next_state
                path.append(state)
                total_reward += reward
                steps += 1
                
                # Visualize
                with grid_container:
                    visualize_grid(env.grid_size, path, state)
                
                with step_container:
                    action_names = ['↑ Up', '↓ Down', '← Left', '→ Right']
                    st.text(f"Step {steps}: {action_names[action]} | Position: {state} | Reward: {reward:.1f}")
                
                time.sleep(0.3)
            
            agent.epsilon = old_epsilon  # Restore epsilon
            
            st.write(f"**Result:** {'🎉 Success!' if done and reward > 0 else '❌ Failed'}")
            st.write(f"**Total Reward:** {total_reward:.2f}")
            st.write(f"**Steps:** {steps}")
            st.write(f"**Path:** {' → '.join([str(p) for p in path])}")
    
    with col2:
        if st.checkbox("Show Q-values (DQN)"):
            st.write("### 🧠 Q-values from Neural Network")
            visualize_q_values(st.session_state.agent_dqn, grid_size)

else:
    st.warning("⚠️ Please train the DQN agent first!")

# Comparison section
st.header("🔄 So sánh Q-Learning vs DQN")

with st.expander("📚 Q-Learning (Q-table) vs DQN (Neural Network)"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Q-Learning (Q-table)")
        st.markdown("""
        **Ưu điểm:**
        - ✅ Đơn giản, dễ hiểu
        - ✅ Guaranteed convergence (với điều kiện phù hợp)
        - ✅ Không cần training neural network
        - ✅ Fast inference
        
        **Nhược điểm:**
        - ❌ Không scale với state space lớn
        - ❌ Chỉ làm việc với discrete states
        - ❌ Cần visit mỗi state nhiều lần
        - ❌ Không có generalization
        
        **Khi nào dùng:**
        - State space nhỏ (< 10,000 states)
        - Discrete states
        - Cần solution đơn giản
        """)
    
    with col2:
        st.subheader("DQN (Neural Network)")
        st.markdown("""
        **Ưu điểm:**
        - ✅ Scale tốt với state space lớn
        - ✅ Generalization: học từ similar states
        - ✅ Có thể làm với continuous states
        - ✅ Experience Replay: học hiệu quả hơn
        
        **Nhược điểm:**
        - ❌ Phức tạp hơn
        - ❌ Cần tune nhiều hyperparameters
        - ❌ Training chậm hơn
        - ❌ Không guaranteed convergence
        
        **Khi nào dùng:**
        - State space lớn (> 10,000 states)
        - Continuous states
        - Cần generalization
        - Có GPU để training
        """)

with st.expander("🔬 Các kỹ thuật trong DQN"):
    st.markdown("""
    ### 1. Experience Replay
    - Lưu trữ experiences (s, a, r, s') trong buffer
    - Sample ngẫu nhiên batch để training
    - **Lợi ích**: Phá vỡ correlation giữa sequential samples
    
    ### 2. Target Network
    - Dùng 2 networks: Policy Network và Target Network
    - Target Network update chậm hơn (mỗi N steps)
    - **Lợi ích**: Training stable hơn, tránh oscillation
    
    ### 3. Epsilon-Greedy với Decay
    - Bắt đầu với epsilon cao (exploration)
    - Giảm dần epsilon theo thời gian
    - **Lợi ích**: Balance exploration và exploitation
    
    ### 4. Gradient Clipping
    - Giới hạn gradient magnitude
    - **Lợi ích**: Tránh exploding gradients
    
    ### Network Architecture
    ```
    Input (2): x, y coordinates (normalized)
    ↓
    Dense(64) + ReLU
    ↓
    Dense(64) + ReLU
    ↓
    Output(4): Q-values cho 4 actions
    ```
    """)