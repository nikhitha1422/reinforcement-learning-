import numpy as np


env_matrix = np.zeros((3, 4))
env_matrix[0, 3] = -1  
env_matrix[1, 3] = -1  
env_matrix[2, 3] = 1   


gamma = 0.8           
alpha = 0.1           
num_episodes = 1000  
max_steps = 100       
epsilon = 0.1         


actions = ['up', 'down', 'left', 'right']


Q = np.zeros((3, 4, len(actions)))


for episode in range(num_episodes):
    state = (0, 0)  
    total_reward = 0
    
    for step in range(max_steps):
        
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(Q[state[0], state[1]])]
        
        
        if action == 'up':
            next_state = (max(state[0] - 1, 0), state[1])
        elif action == 'down':
            next_state = (min(state[0] + 1, 2), state[1])
        elif action == 'left':
            next_state = (state[0], max(state[1] - 1, 0))
        elif action == 'right':
            next_state = (state[0], min(state[1] + 1, 3))
        
        reward = env_matrix[next_state[0], next_state[1]]
        
        
        Q[state[0], state[1], actions.index(action)] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], actions.index(action)])
        
        
        state = next_state
        total_reward += reward
        
        
        if reward == 1 or reward == -1:
            break
    
    
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

print("\nFinal Q-values:")
print(Q)
