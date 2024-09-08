import argparse
import os
import time

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image

from ant import WEIGHTS_DIR, Actor, device, max_abs_reward


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Play Ant environment and optionally save animations."
    )
    parser.add_argument(
        "--save", action="store_true", help="Save episode animations as GIFs"
    )
    args = parser.parse_args()

    # Create gifs folder if saving is enabled
    if args.save:
        gifs_folder = "gifs"
        os.makedirs(gifs_folder, exist_ok=True)

    # Load the environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array" if args.save else "human")

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    # Initialize the actor network
    actor = Actor(state_dim, action_dim, action_low, action_high)

    # Load the trained weights
    actor.load_state_dict(
        torch.load(WEIGHTS_DIR / "td3_ant_actor.pth", map_location=device)
    )
    actor.eval()  # Set the network to evaluation mode

    # Play episodes
    num_episodes = 10
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        frames = []

        while not done:
            state[2] /= 8.0
            # Select action
            state_tensor = torch.Tensor(state).to(device)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()

            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward = reward / max_abs_reward
            done = terminated or truncated

            total_reward += reward
            state = next_state

            # Render and optionally save frame
            frame = env.render()
            if args.save:
                frames.append(Image.fromarray(frame))

            if not args.save:
                time.sleep(0.1)

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

        # Save episode as gif if enabled
        if args.save:
            gif_path = os.path.join(gifs_folder, f"episode_{episode+1}.gif")
            frames[0].save(
                gif_path, save_all=True, append_images=frames[1:], duration=10, loop=0
            )
            print(f"Saved episode animation to {gif_path}")

    env.close()


if __name__ == "__main__":
    main()
