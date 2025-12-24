#!/bin/bash
# Setup script for TTS Backend development environment

set -e

echo "=== TTS Backend Setup ==="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python version: $PYTHON_VERSION"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "✓ Poetry installed"

# Install dependencies
echo "Installing dependencies..."
poetry install
echo "✓ Dependencies installed"

# Setup pre-commit hooks
echo "Setting up pre-commit hooks..."
poetry run pre-commit install
echo "✓ Pre-commit hooks installed"

# Create directories
echo "Creating directories..."
mkdir -p voices
mkdir -p models
mkdir -p logs
echo "✓ Directories created"

# Copy environment file if not exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created (please review and update settings)"
else
    echo "✓ .env file already exists"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Review and update .env file with your settings"
echo "2. Run 'make dev' to start the development server"
echo "3. Visit http://localhost:8000/docs for API documentation"
echo ""
