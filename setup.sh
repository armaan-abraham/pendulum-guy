#!/bin/bash

# Exit on any error
set -e

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

# Install Python 3.11 (or the version specified in .python-version)
PYTHON_VERSION=$(cat .python-version)
if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION="3.11.0"
fi
pyenv install -s $PYTHON_VERSION
pyenv local $PYTHON_VERSION

# Install poetry if not already installed
if ! command -v poetry &> /dev/null; then
    echo "Installing poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Create poetry environment and install dependencies
poetry env use python
poetry install

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

echo "Setup complete!"