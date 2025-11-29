import streamlit as st
import numpy as np
import time
from environment import GridEnvironment
from q_agent import QLearningAgent
from visualization import visualize_grid, visualize_training_stats, visualize_q_values

st.set_page_config(page_title="Q-Learning Grid Navigation", layout="wide")

st.title("🎯 Q-Learning: Grid Navigation")
st.markdown("""
Problem: Find the shortest path from (0,0) to (n-1,n-1) in an n×n grid
- ✅ Movement: Up, Down, Left, Right (1 step at a time)
- ❌ No diagonal movement
- 🚫 Going out of bounds = penalty
""")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

# Grid size
grid_size = st.sidebar.slider("Grid size (n × n)", 3, 15, 5)

# Training parameters
st.sidebar.subheader("Training Parameters")
episodes = st.sidebar.number_input("Episodes", 100, 10000, 1000, 100)
learning_rate = st.sidebar.slider("Learning rate (α)", 0.01, 1.0, 0.1, 0.01)
discount_factor = st.sidebar.slider("Discount factor (γ)", 0.01, 1.0, 0.95, 0.01)
epsilon = st.sidebar.slider("Epsilon (exploration)", 0.0, 1.0, 0.1, 0.01)
epsilon_decay = st.sidebar.slider("Epsilon decay", 0.9, 0.999, 0.995, 0.001)

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
    st.session_state.agent = None
    st.session_state.env = None
    st.session_state.training_history = None

# Create environment and agent
env = GridEnvironment(grid_size)
agent = QLearningAgent(
    state_size=grid_size,
    action_size=4,
    learning_rate=learning_rate,
    discount_factor=discount_factor,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay
)

# Training section
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎓 Train Agent")
    
    if st.button("🚀 Start Training", type="primary"):
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
                status_text.text(f"Episode {episode+1}/{episodes} | Avg Reward: {avg_reward:.2f} | Avg Steps: {avg_steps:.1f} | ε: {agent.epsilon:.3f}")
        
        progress_bar.progress(1.0)
        status_text.success(f"✅ Training completed: {episodes} episodes!")
        
        # Save to session state
        st.session_state.trained = True
        st.session_state.agent = agent
        st.session_state.env = env
        st.session_state.training_history = {
            'rewards': rewards_history,
            'steps': steps_history,
            'epsilon': epsilon_history
        }

with col2:
    st.header("📊 Info")
    if st.session_state.trained:
        st.metric("Status", "✅ Trained")
        history = st.session_state.training_history
        st.metric("Avg Reward (last 100 episodes)", 
                  f"{np.mean(history['rewards'][-100:]):.2f}")
        st.metric("Avg Steps (last 100 episodes)", 
                  f"{np.mean(history['steps'][-100:]):.1f}")
    else:
        st.metric("Status", "⏳ Not trained")

# Training statistics
if st.session_state.trained:
    st.header("📈 Training Statistics")
    visualize_training_stats(st.session_state.training_history)

# Testing section
st.header("🎮 Test Agent")

if st.session_state.trained:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Run 1 Episode", type="secondary"):
            agent = st.session_state.agent
            env = st.session_state.env
            
            state = env.reset()
            path = [state]
            total_reward = 0
            steps = 0
            done = False
            
            st.write("### 🗺️ Path:")
            step_container = st.empty()
            grid_container = st.empty()
            
            agent.epsilon = 0  # Greedy mode for testing
            
            while not done and steps < grid_size * grid_size * 2:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                
                state = next_state
                path.append(state)
                total_reward += reward
                steps += 1
                
                # Visualize current step
                with grid_container:
                    visualize_grid(env.grid_size, path, state)
                
                with step_container:
                    action_names = ['↑ Up', '↓ Down', '← Left', '→ Right']
                    st.text(f"Step {steps}: {action_names[action]} | Position: {state} | Reward: {reward:.1f}")
                
                time.sleep(0.3)  # Delay for visualization
            
            st.write(f"**Result:** {'🎉 Success!' if done and reward > 0 else '❌ Failed'}")
            st.write(f"**Total Reward:** {total_reward:.2f}")
            st.write(f"**Steps:** {steps}")
            st.write(f"**Path:** {' → '.join([str(p) for p in path])}")
    
    with col2:
        if st.checkbox("Show Q-values"):
            st.write("### 🧠 Agent's Q-values")
            visualize_q_values(st.session_state.agent, grid_size)

else:
    st.warning("⚠️ Please train the agent first!")

# Information section
with st.expander("ℹ️ Q-Learning Algorithm Explanation"):
    st.markdown("""
    ### What is Q-Learning?
    
    Q-Learning is a **model-free Reinforcement Learning** algorithm that learns the optimal policy.
    
    ### Q-value Update Formula:
    ```
    Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
    ```
    
    Where:
    - **Q(s,a)**: Value of taking action `a` in state `s`
    - **α**: Learning rate (speed of learning)
    - **r**: Reward received
    - **γ**: Discount factor (weight of future rewards)
    - **s'**: Next state
    
    ### Components:
    1. **State**: Current position (x, y)
    2. **Action**: Up, Down, Left, Right
    3. **Reward**:
       - +100: Reach goal
       - -1: Each step
       - -10: Hit boundary
    4. **Q-table**: Matrix storing Q-values for each (state, action) pair
    
    ### Epsilon-Greedy:
    - **Exploration**: Choose random action with probability ε
    - **Exploitation**: Choose best action with probability 1-ε
    """)

with st.expander("🔄 Transition to Deep Q-Network (DQN)"):
    st.markdown("""
    ### To transition to DQN:
    
    The code is designed in a modular way for easy conversion:
    
    1. **Q-table → Neural Network**:
       - Instead of storing Q-values in a table, use a neural network to approximate them
       - Input: State (x, y)
       - Output: Q-values for all actions
    
    2. **Required Changes**:
       - Create `DQNAgent` class inheriting from `QLearningAgent`
       - Replace `q_table` with neural network (PyTorch/TensorFlow)
       - Add Experience Replay buffer
       - Add Target Network
    
    3. **Advantages of DQN**:
       - Handles large state spaces
       - Works with continuous states
       - Better generalization
    
    Check out `dqn_agent.py` for the implementation!
    """)