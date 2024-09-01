#!/bin/bash

# Exit on any error
set -e

# Define command line arguments
GIT_CONFIG=true
CLONE_REPO=true
INSTALL_PYENV=true
INSTALL_PYTHON=true
INSTALL_POETRY=true
SETUP_ENVIRONMENT=true
CHECK_GPU=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-git-config)
            GIT_CONFIG=false
            shift
            ;;
        --skip-clone-repo)
            CLONE_REPO=false
            shift
            ;;
        --skip-pyenv)
            INSTALL_PYENV=false
            shift
            ;;
        --skip-python)
            INSTALL_PYTHON=false
            shift
            ;;
        --skip-poetry)
            INSTALL_POETRY=false
            shift
            ;;
        --skip-environment)
            SETUP_ENVIRONMENT=false
            shift
            ;;
        --skip-gpu-check)
            CHECK_GPU=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# GitHub configuration
GIT_EMAIL="armaan.abraham@hotmail.com"
REPO_URL="https://github.com/armaan-abraham/recursive-intention.git"

# Store the initial directory
INITIAL_DIR=$(pwd)

if $GIT_CONFIG; then
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
fi

if $CLONE_REPO; then
    # Clone the repository
    git clone https://$GITHUB_TOKEN@github.com/armaan-abraham/recursive-intention.git
    cd recursive-intention
fi

if $INSTALL_PYENV; then
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
fi

if $INSTALL_PYTHON; then
    # Ensure we're in the correct directory
    cd "$INITIAL_DIR/recursive-intention" || exit 1

    # Ask for Python version
    echo "Please enter the Python version you want to use (e.g., 3.11.0):"
    read PYTHON_VERSION

    # Install specified Python version
    pyenv install -s $PYTHON_VERSION
    pyenv local $PYTHON_VERSION

    # Update .python-version file
    echo $PYTHON_VERSION > .python-version
fi

if $INSTALL_POETRY; then
    # Ensure we're in the correct directory
    cd "$INITIAL_DIR/recursive-intention" || exit 1

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
fi

if $SETUP_ENVIRONMENT; then
    # Ensure we're in the correct directory
    cd "$INITIAL_DIR/recursive-intention" || exit 1

    # Create poetry environment and install dependencies
    poetry env use python
    poetry install
fi

if $CHECK_GPU; then
    # Ensure we're in the correct directory
    cd "$INITIAL_DIR/recursive-intention" || exit 1

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