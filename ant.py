import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from tqdm import tqdm
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"

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
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.action_low = torch.FloatTensor(action_low).to(device)
        self.action_high = torch.FloatTensor(action_high).to(device)
        self.to(device)

    def forward(self, state):
        return self.action_high * self.layers(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        # output should be negative
        return self.layers(x)
        # return nn.ReLU()(self.layers(x)) * -1

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_low, action_high):
        # TODO: is this initialization correct?
        self.actor = Actor(state_dim, action_dim, action_low, action_high)
        self.actor_target = Actor(state_dim, action_dim, action_low, action_high)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3, amsgrad=True)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, amsgrad=True)

        self.action_low = torch.FloatTensor(action_low).to(device)
        self.action_high = torch.FloatTensor(action_high).to(device)
        self.noise_scale = 0.1
        self.noise_decay = 0.99

        self.actor_updates_per_critic_update = 1

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_action_with_noise(self, state):
        action = self.select_action(state)
        noise = np.random.normal(0, self.noise_scale, size=action.shape)
        return np.clip(action + noise, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())

    def decay_noise(self):
        self.noise_scale *= self.noise_decay

    def train(self, replay_buffer, iterations, batch_size, discount=0.999, tau=1e-6):
        print("Initial losses:")
        self.print_loss(replay_buffer, batch_size, discount)

        for _ in tqdm(range(iterations), desc="Training", unit="iteration"):
            # Sample from the replay buffer
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)

            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            current_Q = self.critic(state, action)

            self.critic_optimizer.zero_grad()
            critic_loss = nn.functional.mse_loss(current_Q, target_Q)
            critic_loss.backward()
            self.critic_optimizer.step()
            self.soft_update(self.critic, self.critic_target, tau)

            for _ in range(self.actor_updates_per_critic_update):
                self.actor_optimizer.zero_grad()
                actor_loss = -self.critic(state, self.actor(state)).mean()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.soft_update(self.actor, self.actor_target, tau)

        print("Final losses:")
        self.print_loss(replay_buffer, batch_size, discount)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data = (tau * local_param.data + (1.0 - tau) * target_param.data).detach().clone()

    def print_loss(self, replay_buffer, batch_size, discount):
        with torch.no_grad():
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)

            print(f"Average critic target prediction", self.critic_target(state, action).mean().item())
            print(f"critic target prediction", self.critic_target(state, action)[:10])

            # Compute critic loss
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()
            current_Q = self.critic_target(state, action)
            critic_loss = nn.functional.mse_loss(current_Q, target_Q).item()

            print(f"Average actor target prediction", self.actor_target(state).abs().mean().item())
            print(f"actor target prediction", self.actor_target(state)[:10])

            # Compute actor loss
            actor_loss = -self.critic_target(state, self.actor_target(state)).mean().item()

            print(f"Critic loss: {critic_loss:.6f}")
            print(f"Actor loss: {actor_loss:.6f}")

# Define a simple replay buffer
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1e6):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            # TODO: check shapes
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )

REPLAY_BUFFER_BATCH_SIZE = 40000
TRAINING_ITERATIONS = 75
TRAINING_BEGIN = 2

def main():
    num_envs = 256
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
    assert action_low == -action_high

    agent = DDPGAgent(state_dim, action_dim, action_low, action_high)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    total_timesteps = 0
    max_timesteps = 2000000
    episode_num = 0

    max_abs_reward = 16.2736044

    if isinstance(envs.single_observation_space, gym.spaces.Dict):
        state_dict, _ = envs.reset()
        state = state_dict['observation']  # Extract the 'observation' part
    else:
        state, _ = envs.reset()

    state[:, 2] /= 8

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

    flag = False

    while total_timesteps < max_timesteps:
        if total_timesteps <= TRAINING_BEGIN * REPLAY_BUFFER_BATCH_SIZE:
            action = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
        else:
            if not flag:
                print("Starting action...")
                flag = True
            action = np.array([agent.select_action_with_noise(s) for s in state])
            agent.decay_noise()  # Decay the noise after each action selection
        
        next_state_dict, reward, terminated, truncated, info = envs.step(action)
        reward = reward / max_abs_reward
        assert (reward < 0).all()
        if isinstance(envs.single_observation_space, gym.spaces.Dict):  
            next_state = next_state_dict['observation']  # Extract the 'observation' part
        else:
            next_state = next_state_dict
        next_state[:, 2] /= 8
        done = np.logical_or(terminated, truncated)
        # print("next_state:")
        # print(next_state[:10])
        # print("reward:")
        # print(reward[:10])
        # print("action:")
        # print(action[:10])


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
                replay_buffer.add(state[i], action[i], final_obs, reward[i], 1.0)
                total_episodes += 1
                total_episode_reward += episode_rewards[i]
                episode_num += 1
                episode_rewards[i] = 0
                episode_lengths[i] = 0
            else:
                replay_buffer.add(state[i], action[i], next_state[i], reward[i], 0.0)

        state = next_state
        total_timesteps += num_envs
        # pbar.update(num_envs)

        # Update progress bar description
        avg_reward_3000 = np.mean(recent_rewards) if recent_rewards else 0  # Changed from 1000 to 3000
        avg_reward_episode = total_episode_reward / total_episodes if total_episodes > 0 else 0
        # pbar.set_description(f"Reward (3000): {avg_reward_3000:.2f} | Reward (Ep): {avg_reward_episode:.2f} | Episodes: {total_episodes} | Buffer size: {len(replay_buffer.storage):.1e}")

        TRAIN_FREQ = REPLAY_BUFFER_BATCH_SIZE // num_envs

        if iter_count >= TRAIN_FREQ * TRAINING_BEGIN and iter_count % TRAIN_FREQ == 0 and replay_buffer.size >= REPLAY_BUFFER_BATCH_SIZE:
            print("Training...")
            agent.train(replay_buffer, iterations=TRAINING_ITERATIONS, batch_size=REPLAY_BUFFER_BATCH_SIZE)
            print(f"Reward (3000): {avg_reward_3000:.2f} | Reward (Ep): {avg_reward_episode:.2f} | Episodes: {total_episodes} | Buffer size: {replay_buffer.size:.1e}")
            print("Noise scale: ", agent.noise_scale)

        SAVE_FREQ = 500

        if iter_count % SAVE_FREQ == 0:
            torch.save(agent.actor.state_dict(), WEIGHTS_DIR / f"ddpg_ant_actor_{iter_count}.pth")
            torch.save(agent.critic.state_dict(), WEIGHTS_DIR / f"ddpg_ant_critic_{iter_count}.pth")

        iter_count += 1

    # save model
    torch.save(agent.actor.state_dict(), WEIGHTS_DIR / "ddpg_ant_actor.pth")
    torch.save(agent.critic.state_dict(), WEIGHTS_DIR / "ddpg_ant_critic.pth")

    pbar.close()

if __name__ == "__main__":
    main()
