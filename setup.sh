#!/bin/bash

# Exit on any error
set -e

# Define steps
STEP_GIT=false
STEP_PYTHON=false
STEP_POETRY=false
STEP_GPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --git)
            STEP_GIT=true
            shift
            ;;
        --python)
            STEP_PYTHON=true
            shift
            ;;
        --poetry)
            STEP_POETRY=true
            shift
            ;;
        --gpu)
            STEP_GPU=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--git] [--python] [--poetry] [--gpu]"
            exit 1
            ;;
    esac
done

# GitHub configuration
GIT_EMAIL="armaan.abraham@hotmail.com"
REPO_URL="https://github.com/armaan-abraham/recursive-intention.git"

# Store the initial directory
INITIAL_DIR=$(pwd)

if $STEP_GIT; then
    echo "Running Git configuration and repository clone..."
    
    # Configure Git
    git config --global user.email "$GIT_EMAIL"
    echo "Git email set to $GIT_EMAIL"

    # Ask for GitHub access token
    echo "Please enter your GitHub access token:"
    read -s GITHUB_TOKEN

    # Test the token
    if curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user | grep -q "login"; then
        echo "GitHub authentication successful"
    else
        echo "GitHub authentication failed. Please check your token and try again."
        exit 1
    fi

    # Clone the repository
    git clone https://$GITHUB_TOKEN@github.com/armaan-abraham/recursive-intention.git
    cd recursive-intention
fi

if $STEP_PYTHON; then
    echo "Installing pyenv and Python..."
    
    # Install pyenv if not already installed
    if ! command -v pyenv &> /dev/null; then
        echo "Installing pyenv..."
        curl https://pyenv.run | bash
        
        # Add pyenv to PATH
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        
        # Reload shell
        exec "$SHELL"
    fi

    # Ask for Python version
    echo "Please enter the Python version you want to use (e.g., 3.11.0):"
    read PYTHON_VERSION

    # Install specified Python version
    pyenv install -s $PYTHON_VERSION
    pyenv local $PYTHON_VERSION

    # Update .python-version file
    echo $PYTHON_VERSION > .python-version
fi

if $STEP_POETRY; then
    echo "Installing Poetry and setting up environment..."
    
    # Install poetry if not already installed
    if ! command -v poetry &> /dev/null; then
        echo "Installing poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        
        # Add Poetry to PATH
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
    fi

    # Ensure Poetry is available in the current session
    export PATH="$HOME/.local/bin:$PATH"

    # Check if Poetry is now available
    if command -v poetry &> /dev/null; then
        echo "Poetry installed successfully"
    else
        echo "Poetry installation failed. Please install manually and rerun the script."
        exit 1
    fi

    # Create poetry environment and install dependencies
    poetry env use python
    poetry install
fi

if $STEP_GPU; then
    echo "Checking GPU..."

    # Check for CUDA GPU
    echo "Checking for CUDA GPU..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
    else
        echo "WARNING: NVIDIA GPU not detected or nvidia-smi not found."
    fi

    # Check if PyTorch can access GPU
    echo "Checking PyTorch GPU access..."
    poetry run python - <<EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: PyTorch cannot access GPU.")
EOF
fi

echo "Setup complete!"