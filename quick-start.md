# 🚀 Quick Start Guide

Get started with Q-Learning and DQN in 5 minutes!

## ⚡ Installation (1 minute)

```bash
pip install streamlit numpy pandas plotly torch
```

## 🎮 Run Applications (1 minute)

### Option 1: Q-Learning (Recommended for beginners)
```bash
streamlit run app.py
```

### Option 2: DQN (Advanced)
```bash
streamlit run app_dqn.py
```

### Option 3: Quick Test
```bash
python test_agents.py
```

## 🎯 First Training (3 minutes)

1. **Open browser** at `http://localhost:8501`

2. **Configure parameters** in sidebar:
   - Grid size: **5** (start small)
   - Episodes: **1000**
   - Learning rate: **0.1** (Q-Learning) or **0.001** (DQN)
   - Epsilon: **0.1**

3. **Click "Start Training"**
   - Watch progress bar
   - Monitor metrics
   - Wait ~30 seconds

4. **Test the agent**
   - Click "Run 1 Episode"
   - Watch step-by-step visualization
   - See the learned path

## 📊 Understanding Results

### Training Charts

**Rewards Chart**:
- ↗️ Increasing trend = Learning successfully
- → Stable at high value = Converged
- ↘️ Decreasing = Problem with parameters

**Steps Chart**:
- ↘️ Decreasing trend = Finding shorter paths
- → Stable = Found optimal path
- ↗️ Increasing = Agent struggling

### Test Results

- **Success**: Agent reaches goal
- **Steps ≈ 2(n-1)**: Near-optimal (straight path is n-1 + n-1)
- **High reward**: Good path efficiency

## 🎓 What You'll Learn

### From Q-Learning
- ✅ Reinforcement Learning basics
- ✅ Q-table and Q-values
- ✅ Epsilon-greedy exploration
- ✅ Bellman equation
- ✅ Hyperparameter tuning

### From DQN
- ✅ Deep Reinforcement Learning
- ✅ Neural networks for RL
- ✅ Experience Replay
- ✅ Target Networks
- ✅ Scaling RL to larger problems

## 🔧 Recommended Settings

### First Run (Easy)
```
Grid: 5×5
Episodes: 1000
Learning rate: 0.1 (Q-Learning) / 0.001 (DQN)
Epsilon: 0.1
```

### After Understanding (Challenging)
```
Grid: 10×10
Episodes: 2000
Experiment with different parameters
```

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `pip install -r requirements.txt` |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Agent not learning | Increase episodes or reduce epsilon |
| Training too slow | Reduce grid size or episodes |

## 📚 Next Steps

1. ✅ Run Q-Learning successfully
2. ✅ Understand the training charts
3. ✅ Run DQN and compare
4. ✅ Experiment with parameters
5. ✅ Read full [README.md](README.md) for details

## 💡 Pro Tips

1. **Start simple**: 5×5 grid, 1000 episodes
2. **Visualize**: Use apps to see agent learn
3. **Compare**: Run both Q-Learning and DQN
4. **Experiment**: Change one parameter at a time
5. **Understand**: Read the code after seeing it work

---

**Ready?** Let's start learning!

```bash
streamlit run app.py
```

For detailed documentation, see [README.md](README.md)