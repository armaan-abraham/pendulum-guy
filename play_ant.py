import numpy as np
import torch
import gymnasium as gym
from ant import Actor, device, WEIGHTS_DIR

def main():
    # Load the environment
    env = gym.make("Pendulum-v1", render_mode="human")
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    # Initialize the actor network
    actor = Actor(state_dim, action_dim, action_low, action_high)
    
    # Load the trained weights
    actor.load_state_dict(torch.load(WEIGHTS_DIR / "ddpg_ant_actor.pth", map_location=device))
    actor.eval()  # Set the network to evaluation mode

    # Play episodes
    num_episodes = 5
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy().flatten()
            
            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state
            
            env.render()
        
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()