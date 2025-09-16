#!/bin/bash
# postCreateCommand.sh - DevContainer環境専用（Python + Node.js）

set -e

echo "START DevContainer Setup (Python + Node.js)"

# Initialize firewall first
echo "Setting up firewall..."
sudo /usr/local/bin/init-firewall.sh

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the rest of this script's execution
    if [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
fi

# 権限設定
sudo chown -R node:node .

# 作業ディレクトリを設定
cd "${containerWorkspaceFolder}"

# Check if this is a Python project (has pyproject.toml)
if [ -f "pyproject.toml" ]; then
    echo "Python project detected. Setting up Python environment..."

    # 既存の仮想環境を削除（権限問題を回避するため）
    if [ -d ".venv" ]; then
        echo "Removing existing virtual environment..."
        sudo rm -rf .venv
    fi

    # 新しい仮想環境を作成
    echo "Creating Python virtual environment..."
    uv venv --python 3.12
    
    # 仮想環境の権限を設定
    sudo chown -R node:node .venv

    # 仮想環境を有効化
    source .venv/bin/activate

    # 依存関係をインストール
    echo "Installing Python dependencies..."
    uv sync

    # pre-commit設定（存在する場合）
    if uv pip list | grep -q pre-commit; then
        echo "Setting up pre-commit..."
        uv run -- pre-commit install
    fi
else
    echo "No pyproject.toml found. Skipping Python environment setup."
fi

# Check if this is a Node.js project (has package.json)
if [ -f "package.json" ]; then
    echo "Node.js project detected. Installing npm dependencies..."
    npm install
else
    echo "No package.json found. Skipping npm dependencies."
fi

# Git設定
echo "Setting up Git configuration..."
if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then
    git config --global user.name "$GIT_USER_NAME"
    git config --global user.email "$GIT_USER_EMAIL"
else
    echo "GIT_USER_NAME and/or GIT_USER_EMAIL not set. Using default git config."
    git config --global user.name "DevContainer User"
    git config --global user.email "devcontainer@example.com"
fi
git config --global init.defaultBranch main

# Create useful aliases
echo "Setting up shell aliases..."
cat >> ~/.zshrc << 'EOF'

# Project aliases
alias ll='ls -la'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Python aliases
alias py='python'
alias pip='uv pip'
alias venv='source .venv/bin/activate'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'

# AI CLI aliases
alias claude='claude-code'
alias gem='gemini'

# Development aliases
alias runserver='python -m http.server 8000'
alias jnb='jupyter notebook --ip=0.0.0.0 --no-browser --allow-root'

# Activate Python virtual environment automatically
if [ -d ".venv/bin" ]; then
  echo 'export PATH="/workspace/.venv/bin:$PATH"' >> ~/.zshrc
fi

EOF

echo "FINISH DevContainer Setup (Python + Node.js)"
