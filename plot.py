from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ant import WEIGHTS_DIR, Actor, device

PLOT_DIR = Path(__file__).parent / "plots"


def plot_combined_weights(actor):
    # Get all layers with weights
    weight_layers = [layer for layer in actor.layers if hasattr(layer, "weight")]

    # Combine weights from second to second-to-last layer
    combined_weights = [
        layer.weight.data.cpu().numpy() for layer in weight_layers[1:-1]
    ]
    combined_weights = [np.log(np.abs(weights)) for weights in combined_weights]

    # Calculate the total width and individual layer widths
    layer_widths = [layer.shape[1] for layer in combined_weights]

    # Concatenate the weights
    combined_weights = np.concatenate(combined_weights, axis=1)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 10))

    # Create the heatmap
    sns.heatmap(
        combined_weights,
        center=0,
        annot=False,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Weight Value"},
    )

    # Add vertical lines to separate layers
    cumulative_width = 0
    for i, width in enumerate(layer_widths):
        cumulative_width += width
        if i < len(layer_widths) - 1:  # Don't draw a line after the last layer
            ax.axvline(x=cumulative_width, color="red", linestyle="--")

        # Add layer number label
        ax.text(
            cumulative_width - width / 2, -0.5, f"Layer {i+2}", ha="center", va="top"
        )

    # Set the title and labels
    ax.set_title("Combined Weights Heatmap (2nd to 2nd-to-last layer)")
    ax.set_xlabel("Neurons (Output) by Layer")
    ax.set_ylabel("Neurons (Input)")

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "combined_weights.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Load the environment to get dimensions
    env = gym.make("Pendulum-v1")

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

    # Plot combined weights
    plot_combined_weights(actor)

    env.close()


if __name__ == "__main__":
    main()
