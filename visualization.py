import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def visualize_grid(grid_size, path=None, current_pos=None):
    """
    Display grid with path
    
    Args:
        grid_size: Grid size
        path: List of visited positions
        current_pos: Current position
    """
    # Create display matrix
    grid = np.zeros((grid_size, grid_size))
    
    # Mark path
    if path:
        for i, pos in enumerate(path):
            x, y = pos
            grid[x, y] = i + 1
    
    # Tạo color scale
    colors = []
    annotations = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Background color
            if (i, j) == (0, 0):
                color = 'lightgreen'
                text = '🏁 START'
            elif (i, j) == (grid_size-1, grid_size-1):
                color = 'lightcoral'
                text = '🎯 GOAL'
            elif current_pos and (i, j) == current_pos:
                color = 'yellow'
                text = '🤖'
            elif path and (i, j) in path:
                step_num = path.index((i, j))
                color = f'rgba(100, 149, 237, {0.3 + 0.7 * step_num / len(path)})'
                text = str(step_num + 1)
            else:
                color = 'white'
                text = ''
            
            colors.append(color)
            annotations.append(dict(
                x=j, y=i,
                text=text,
                showarrow=False,
                font=dict(size=14, color='black')
            ))
    
    # Reshape colors thành grid
    color_grid = np.array(colors).reshape(grid_size, grid_size)
    
    # Tạo heatmap
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        colorscale=[[0, 'white'], [1, 'cornflowerblue']],
        showscale=False,
        hovertemplate='Position: (%{x}, %{y})<extra></extra>'
    ))
    
    # Add grid lines
    for i in range(grid_size + 1):
        fig.add_shape(
            type="line",
            x0=-0.5, y0=i-0.5, x1=grid_size-0.5, y1=i-0.5,
            line=dict(color="gray", width=1)
        )
        fig.add_shape(
            type="line",
            x0=i-0.5, y0=-0.5, x1=i-0.5, y1=grid_size-0.5,
            line=dict(color="gray", width=1)
        )
    
    # Add annotations
    fig.update_layout(
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, autorange='reversed'),
        width=500,
        height=500,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=False)

def visualize_training_stats(history):
    """
    Display training statistics
    
    Args:
        history: Dictionary chứa rewards, steps, epsilon
    """
    col1, col2 = st.columns(2)
    
    with col1:
        # Rewards over time
        fig_reward = go.Figure()
        
        # Raw rewards
        fig_reward.add_trace(go.Scatter(
            y=history['rewards'],
            mode='lines',
            name='Reward',
            line=dict(color='lightblue', width=1),
            opacity=0.3
        ))
        
        # Moving average
        window = 100
        if len(history['rewards']) >= window:
            moving_avg = pd.Series(history['rewards']).rolling(window=window).mean()
            fig_reward.add_trace(go.Scatter(
                y=moving_avg,
                mode='lines',
                name=f'Moving Avg ({window})',
                line=dict(color='blue', width=2)
            ))
        
        fig_reward.update_layout(
            title='Rewards per Episodes',
            xaxis_title='Episode',
            yaxis_title='Total Reward',
            height=300
        )
        st.plotly_chart(fig_reward, use_container_width=True)
    
    with col2:
        # Steps over time
        fig_steps = go.Figure()
        
        # Raw steps
        fig_steps.add_trace(go.Scatter(
            y=history['steps'],
            mode='lines',
            name='Steps',
            line=dict(color='lightcoral', width=1),
            opacity=0.3
        ))
        
        # Moving average
        if len(history['steps']) >= window:
            moving_avg = pd.Series(history['steps']).rolling(window=window).mean()
            fig_steps.add_trace(go.Scatter(
                y=moving_avg,
                mode='lines',
                name=f'Moving Avg ({window})',
                line=dict(color='red', width=2)
            ))
        
        fig_steps.update_layout(
            title='Steps per Episodes',
            xaxis_title='Episode',
            yaxis_title='Steps',
            height=300
        )
        st.plotly_chart(fig_steps, use_container_width=True)
    
    # Epsilon decay
    fig_epsilon = go.Figure()
    fig_epsilon.add_trace(go.Scatter(
        y=history['epsilon'],
        mode='lines',
        name='Epsilon',
        line=dict(color='green', width=2)
    ))
    
    fig_epsilon.update_layout(
        title='Epsilon (Exploration Rate) per Episodes',
        xaxis_title='Episode',
        yaxis_title='Epsilon',
        height=250
    )
    st.plotly_chart(fig_epsilon, use_container_width=True)

def visualize_q_values(agent, grid_size):
    """
    Display Q-values as arrows for each state
    
    Args:
        agent: Q-learning agent
        grid_size: Kích thước grid
    """
    # Tạo grid để hiển thị
    fig = go.Figure()
    
    # Action directions for arrows
    action_deltas = {
        0: (0, 0.3),    # Up
        1: (0, -0.3),   # Down
        2: (-0.3, 0),   # Left
        3: (0.3, 0)     # Right
    }
    
    action_colors = {
        0: 'red',    # Up
        1: 'blue',   # Down
        2: 'green',  # Left
        3: 'orange'  # Right
    }
    
    # Vẽ arrows cho mỗi state
    for i in range(grid_size):
        for j in range(grid_size):
            state = (i, j)
            q_values = agent.get_q_values(state)
            best_action = np.argmax(q_values)
            
            # Vẽ arrow cho best action
            dx, dy = action_deltas[best_action]
            
            fig.add_annotation(
                x=j, y=i,
                ax=j + dx, ay=i + dy,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=action_colors[best_action]
            )
            
            # Hiển thị Q-value
            max_q = q_values[best_action]
            fig.add_annotation(
                x=j, y=i,
                text=f'{max_q:.1f}',
                showarrow=False,
                font=dict(size=10, color='black')
            )
    
    # Đánh dấu start và goal
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=20, color='lightgreen', symbol='square'),
        name='Start',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[grid_size-1], y=[grid_size-1],
        mode='markers',
        marker=dict(size=20, color='lightcoral', symbol='star'),
        name='Goal',
        showlegend=True
    ))
    
    # Grid lines
    for i in range(grid_size + 1):
        fig.add_shape(
            type="line",
            x0=-0.5, y0=i-0.5, x1=grid_size-0.5, y1=i-0.5,
            line=dict(color="gray", width=1)
        )
        fig.add_shape(
            type="line",
            x0=i-0.5, y0=-0.5, x1=i-0.5, y1=grid_size-0.5,
            line=dict(color="gray", width=1)
        )
    
    fig.update_layout(
        title='Q-values and Best Actions (Arrows)',
        xaxis=dict(range=[-0.5, grid_size-0.5], showgrid=False),
        yaxis=dict(range=[-0.5, grid_size-0.5], showgrid=False, autorange='reversed'),
        width=600,
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=False)