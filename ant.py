import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from tqdm import tqdm

# Add this at the beginning of the file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU for training.")
else:
    print("WARNING: GPU not available. Using CPU for training. This may be slower.")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim),
        )
        self.action_low = torch.FloatTensor(action_low).to(device)
        self.action_high = torch.FloatTensor(action_high).to(device)
        self.to(device)

    def forward(self, state):
        x = self.layers(state)
        return self.action_low + (self.action_high - self.action_low) * torch.sigmoid(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        self.to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.layers(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_low, action_high):
        self.actor = Actor(state_dim, action_dim, action_low, action_high)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5, amsgrad=True)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-5, amsgrad=True)

        self.action_low = torch.FloatTensor(action_low).to(device)
        self.action_high = torch.FloatTensor(action_high).to(device)
        self.noise_scale = 0.05
        self.noise_decay = 0.98
        self.actor_updates_per_critic_update = 10

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_action_with_noise(self, state):
        action = self.select_action(state)
        noise = np.random.normal(0, self.noise_scale, size=action.shape)
        return np.clip(action + noise, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())

    def decay_noise(self):
        self.noise_scale *= self.noise_decay

    def train(self, replay_buffer, iterations, batch_size, discount=0.99, tau=1e-5):
        print("Initial losses:")
        print(f"Critic loss: {self.get_critic_loss(replay_buffer, batch_size, discount)}")
        print(f"Actor loss: {self.get_actor_loss(replay_buffer, batch_size)}")

        for _ in range(iterations):
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)
            
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.FloatTensor(reward).reshape(-1, 1).to(device)
            done = torch.FloatTensor(done).reshape(-1, 1).to(device)

            target_Q = self.critic_target(next_state, self.actor(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            current_Q = self.critic(state, action)

            critic_loss = nn.functional.mse_loss(current_Q, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            for _ in range(self.actor_updates_per_critic_update):
                actor_loss = -self.critic_target(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

        print("Final losses:")
        print(f"Critic loss: {self.get_critic_loss(replay_buffer, batch_size, discount)}")
        print(f"Actor loss: {self.get_actor_loss(replay_buffer, batch_size)}")

    def get_critic_loss(self, replay_buffer, batch_size, discount):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(device)

        target_Q = self.critic_target(next_state, self.actor(next_state))
        target_Q = reward + (done * discount * target_Q).detach()
        current_Q = self.critic(state, action)
        return nn.functional.mse_loss(current_Q, target_Q).item()

    def get_actor_loss(self, replay_buffer, batch_size):
        state, _, _, _, _ = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        return -self.critic_target(state, self.actor(state)).mean().item()

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
    num_envs = 128
    envs = AsyncVectorEnv([lambda: gym.make("Pendulum-v1") for i in range(num_envs)])
    envs.reset(seed=42)
    
    # Extract the 'observation' part of the space
    if isinstance(envs.single_observation_space, gym.spaces.Box):
        observation_space = envs.single_observation_space
    else:
        observation_space = envs.single_observation_space['observation']
    
    print(observation_space)

    state_dim = observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    action_low = envs.single_action_space.low
    action_high = envs.single_action_space.high

    agent = DDPGAgent(state_dim, action_dim, action_low, action_high)
    replay_buffer = ReplayBuffer()

    total_timesteps = 0
    max_timesteps = 2000000
    episode_num = 0

    max_abs_reward = 16.2736044

    if isinstance(envs.single_observation_space, gym.spaces.Dict):
        state_dict, _ = envs.reset()
        state = state_dict['observation']  # Extract the 'observation' part
    else:
        state, _ = envs.reset()

    episode_rewards = [0] * num_envs
    episode_lengths = [0] * num_envs

    print(f"Starting training with {num_envs} environments") 
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}") 
    print(f"Action space: Low {action_low}, High {action_high}")

    pbar = tqdm(total=max_timesteps, desc="Training Progress")
    recent_rewards = []
    total_episodes = 0
    total_episode_reward = 0
    iter_count = 0

    while total_timesteps < max_timesteps:
        if total_timesteps < 50000:
            action = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
        else:
            action = np.array([agent.select_action_with_noise(s) for s in state])
            agent.decay_noise()  # Decay the noise after each action selection
        
        next_state_dict, reward, terminated, truncated, info = envs.step(action)
        reward = reward / max_abs_reward
        assert (reward < 0).all()
        if isinstance(envs.single_observation_space, gym.spaces.Dict):  
            next_state = next_state_dict['observation']  # Extract the 'observation' part
        else:
            next_state = next_state_dict
        done = np.logical_or(terminated, truncated)

        for i in range(num_envs):
            episode_rewards[i] += reward[i]
            episode_lengths[i] += 1
            recent_rewards.append(reward[i])
            if len(recent_rewards) > 3000:  # Changed from 1000 to 3000
                recent_rewards.pop(0)

            if done[i]:
                if isinstance(info['final_observation'][i], dict):
                    final_obs = info['final_observation'][i]['observation']
                else:
                    final_obs = info['final_observation'][i]
                replay_buffer.add((state[i], action[i], final_obs, reward[i], 1.0))
                total_episodes += 1
                total_episode_reward += episode_rewards[i]
                episode_num += 1
                episode_rewards[i] = 0
                episode_lengths[i] = 0
            else:
                replay_buffer.add((state[i], action[i], next_state[i], reward[i], 0.0))

        state = next_state
        total_timesteps += num_envs
        # pbar.update(num_envs)

        # Update progress bar description
        avg_reward_3000 = np.mean(recent_rewards) if recent_rewards else 0  # Changed from 1000 to 3000
        avg_reward_episode = total_episode_reward / total_episodes if total_episodes > 0 else 0
        # pbar.set_description(f"Reward (3000): {avg_reward_3000:.2f} | Reward (Ep): {avg_reward_episode:.2f} | Episodes: {total_episodes} | Buffer size: {len(replay_buffer.storage):.1e}")

        REPLAY_BUFFER_BATCH_SIZE = 20000
        TRAIN_FREQ = 100

        if iter_count % TRAIN_FREQ == 0:
            lookback = min(REPLAY_BUFFER_BATCH_SIZE, len(replay_buffer.storage))
            if lookback > 0:
                recent_rewards = [exp[3] for exp in replay_buffer.storage[-lookback:]]  # Get rewards from last 1000 experiences
            agent.train(replay_buffer, iterations=min(100, lookback // 100), batch_size=REPLAY_BUFFER_BATCH_SIZE)
            print(f"Reward (3000): {avg_reward_3000:.2f} | Reward (Ep): {avg_reward_episode:.2f} | Episodes: {total_episodes} | Buffer size: {len(replay_buffer.storage):.1e}")
            print("Noise scale: ", agent.noise_scale)

        SAVE_FREQ = 1000
        

        iter_count += 1

    # save model
    torch.save(agent.actor.state_dict(), "ddpg_ant_actor.pth")
    torch.save(agent.critic.state_dict(), "ddpg_ant_critic.pth")

    pbar.close()

if __name__ == "__main__":
    main()
