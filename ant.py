import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.layer_1(state))
        a = torch.relu(self.layer_2(a))
        return self.max_action * torch.tanh(self.layer_3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.cat([state, action], 1)
        q = torch.relu(self.layer_1(q))
        q = torch.relu(self.layer_2(q))
        return self.layer_3(q)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        for _ in range(iterations):
            # Sample a batch of transitions from the replay buffer
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)
            
            # Convert to tensor
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor(reward).reshape(-1, 1)
            done = torch.FloatTensor(done).reshape(-1, 1)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = nn.functional.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Define a simple replay buffer
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind: 
            state, action, next_state, reward, done = self.storage[i]
            batch_states.append(np.asarray(state))
            batch_actions.append(np.asarray(action))
            batch_next_states.append(np.asarray(next_state))
            batch_rewards.append(np.asarray(reward))
            batch_dones.append(np.asarray(done))
        return np.array(batch_states), np.array(batch_actions), np.array(batch_next_states), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

def main():
    env = gym.make('PointMaze_UMaze-v3')
    print(env.observation_space)
    
    # Extract the 'observation' part of the space
    observation_space = env.observation_space['observation']
    state_dim = observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    total_timesteps = 0
    max_timesteps = 1000000
    episode_num = 0

    while total_timesteps < max_timesteps:
        episode_reward = 0
        episode_timesteps = 0
        done = False
        state_dict, _ = env.reset()
        state = state_dict['observation']  # Extract the 'observation' part

        while not done:
            if total_timesteps < 10000:
                action = env.action_space.sample()
            else:
                action = agent.select_action(np.array(state))
                noise = np.random.normal(0, max_action * 0.1, size=action_dim)
                action = (action + noise).clip(-max_action, max_action)

            next_state_dict, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state_dict['observation']  # Extract the 'observation' part
            done = terminated or truncated
            episode_reward += reward

            replay_buffer.add((state, action, next_state, reward, float(done)))

            state = next_state
            episode_timesteps += 1
            total_timesteps += 1

            if total_timesteps % 50 == 0:
                agent.train(replay_buffer, iterations=50)

        episode_num += 1
        print(f"Total Timesteps: {total_timesteps}, Episode Num: {episode_num}, Episode Reward: {episode_reward}")

if __name__ == "__main__":
    main()