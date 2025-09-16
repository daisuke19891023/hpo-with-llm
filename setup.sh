#!/bin/bash

set -e

# Function to print error messages
error() {
    echo "❌ エラー: $1" >&2
    exit 1
}

# Function to print success messages
success() {
    echo "✅ 成功: $1"
}

# Function to print info messages
info() {
    echo "ℹ️ 情報: $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# --- License Text Functions ---
if [ -f "./license_utils.sh" ]; then
    source ./license_utils.sh
else
    # Basic license functions if license_utils.sh doesn't exist
    get_mit_license_text() {
        cat << 'EOF'
MIT License

Copyright (c) {{YEAR}} {{COPYRIGHT_HOLDER}}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
    }
fi

# Banner
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 Python Project Setup Script                 ║"
echo "║              Powered by Claude Code DevContainer            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in a DevContainer
if [ "$DEVCONTAINER" != "true" ]; then
    info "DevContainer環境外で実行されています。uv等のツールが利用可能か確認してください。"
fi

# Check for required project files
SETUP_TYPE=""
if [ -f "pyproject.toml" ] && [ -f "README.md" ]; then
    SETUP_TYPE="update"
    info "既存のPythonプロジェクトが検出されました。設定を更新します。"
elif [ -f "pyproject.toml" ]; then
    SETUP_TYPE="partial"
    info "pyproject.tomlが存在します。部分的なセットアップを行います。"
else
    SETUP_TYPE="new"
    info "新しいプロジェクトのセットアップを開始します。"
fi

# --- Interactive Prompts ---
echo ""
info "プロジェクト情報を入力してください:"
read -p "📦 プロジェクト名: " PROJECT_NAME
read -p "📝 プロジェクトの説明: " PROJECT_DESCRIPTION

# Input validation for LIBRARY_NAME
while true; do
    read -p "🐍 src/ ディレクトリのベースライブラリ名 (例: my_library): " LIBRARY_NAME
    if [[ -z "$LIBRARY_NAME" ]]; then
        echo "❌ エラー: ライブラリ名は空にできません。"
    elif ! [[ "$LIBRARY_NAME" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
        echo "❌ エラー: ライブラリ名が無効です。英数字とアンダースコアのみを使用し、先頭は英字またはアンダースコアにしてください。"
    else
        break
    fi
done

# License selection
echo ""
info "📜 ライセンスタイプを選択してください:"
select LICENSE_TYPE_CHOICE in "MIT" "Apache-2.0" "GPL-3.0" "スキップ"; do
    if [[ -n "$LICENSE_TYPE_CHOICE" ]]; then
        LICENSE_TYPE=$LICENSE_TYPE_CHOICE
        break
    else
        echo "❌ 無効な選択です。もう一度選択してください。"
    fi
done

if [ "$LICENSE_TYPE" != "スキップ" ]; then
    read -p "©️ 著作権者: " COPYRIGHT_HOLDER
fi

# Python version selection
echo ""
info "🐍 Python バージョンを選択してください:"
select PYTHON_VERSION in "3.13" "3.12" "3.11" "3.10" "システムデフォルト"; do
    if [[ -n "$PYTHON_VERSION" ]]; then
        if [ "$PYTHON_VERSION" == "システムデフォルト" ]; then
            PYTHON_VERSION=""
        fi
        break
    else
        echo "❌ 無効な選択です。もう一度選択してください。"
    fi
done

# --- Generate PROJECT_NAME_SLUG ---
info "PROJECT_NAME_SLUGを生成しています..."
PROJECT_NAME_SLUG=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | sed -e 's/[^a-z0-9-]/-/g' -e 's/-\{2,\}/-/g' -e 's/^-//' -e 's/-$//')
success "PROJECT_NAME_SLUGが生成されました: $PROJECT_NAME_SLUG"

# --- Create Library Directory ---
info "ライブラリディレクトリを作成しています..."
mkdir -p "src/$LIBRARY_NAME"
touch "src/$LIBRARY_NAME/__init__.py"

# Create main.py file for the library
cat > "src/$LIBRARY_NAME/main.py" << EOF
#!/usr/bin/env python3
"""Main entry point for $PROJECT_NAME."""

import sys
from typing import NoReturn


def main() -> NoReturn:
    """Main function."""
    print("Hello from $PROJECT_NAME!")
    sys.exit(0)


if __name__ == "__main__":
    main()
EOF

success "ライブラリディレクトリ src/$LIBRARY_NAME が作成されました"

# --- Move Utils Directory ---
# Check for utils in root directory
if [ -d "utils" ]; then
    info "utilsディレクトリをライブラリ配下に移動しています..."

    # Create utils directory in the library
    mkdir -p "src/$LIBRARY_NAME/utils"

    # Copy utils files to the library directory
    cp -r utils/* "src/$LIBRARY_NAME/utils/"

    # Remove the original utils directory
    rm -rf utils

    success "utilsディレクトリが src/$LIBRARY_NAME/utils に移動されました"
# Check for utils in test_project directory
elif [ -d "src/test_project/utils" ]; then
    info "test_project/utilsディレクトリをライブラリ配下に移動しています..."

    # Create utils directory in the library if it doesn't exist
    mkdir -p "src/$LIBRARY_NAME/utils"

    # Copy utils files to the library directory
    cp -r src/test_project/utils/* "src/$LIBRARY_NAME/utils/"

    # Update import paths in utils files
    find "src/$LIBRARY_NAME/utils" -type f -name "*.py" -exec sed -i "s/from test_project/from $LIBRARY_NAME/g; s/import test_project/import $LIBRARY_NAME/g" {} \;

    success "utilsディレクトリが src/$LIBRARY_NAME/utils に移動されました"
else
    info "utilsディレクトリが見つかりません。スキップします。"
fi

# --- Move Utils Tests ---
# Check for tests in unit/test_logger.py location
if [ -d "tests/unit" ] && [ -f "tests/unit/test_logger.py" ]; then
    info "utilsテストファイルをライブラリ配下に移動しています..."

    # Create tests directory structure in the library
    mkdir -p "tests/unit/$LIBRARY_NAME/utils"
    mkdir -p "tests/e2e/$LIBRARY_NAME/utils"

    # Move logger tests to the appropriate location
    if [ -f "tests/unit/test_logger.py" ]; then
        mv "tests/unit/test_logger.py" "tests/unit/$LIBRARY_NAME/utils/"
        success "test_logger.pyが tests/unit/$LIBRARY_NAME/utils/ に移動されました"
    fi

    if [ -f "tests/e2e/test_logging_integration.py" ]; then
        mv "tests/e2e/test_logging_integration.py" "tests/e2e/$LIBRARY_NAME/utils/"
        success "test_logging_integration.pyが tests/e2e/$LIBRARY_NAME/utils/ に移動されました"
    fi

    # Update import paths in test files
    if [ -f "tests/unit/$LIBRARY_NAME/utils/test_logger.py" ]; then
        sed -i "s|from utils.logger|from $LIBRARY_NAME.utils.logger|g" "tests/unit/$LIBRARY_NAME/utils/test_logger.py"
        info "test_logger.pyのインポートパスを更新しました"
    fi

    if [ -f "tests/e2e/$LIBRARY_NAME/utils/test_logging_integration.py" ]; then
        sed -i "s|from utils.logger|from $LIBRARY_NAME.utils.logger|g" "tests/e2e/$LIBRARY_NAME/utils/test_logging_integration.py"
        info "test_logging_integration.pyのインポートパスを更新しました"
    fi

    success "utilsテストファイルの移動と更新が完了しました"
# Check for tests in test_project directory structure
elif [ -d "tests/unit/test_project/utils" ] || [ -d "tests/e2e/test_project/utils" ]; then
    info "test_project/utilsテストファイルをライブラリ配下に移動しています..."

    # Create tests directory structure in the library
    mkdir -p "tests/unit/$LIBRARY_NAME/utils"
    mkdir -p "tests/e2e/$LIBRARY_NAME/utils"
    touch "tests/unit/$LIBRARY_NAME/__init__.py"
    touch "tests/e2e/$LIBRARY_NAME/__init__.py"

    # Move unit tests
    if [ -d "tests/unit/test_project/utils" ]; then
        cp -r tests/unit/test_project/utils/* "tests/unit/$LIBRARY_NAME/utils/"
        # Update import paths in test files
        find "tests/unit/$LIBRARY_NAME/utils" -type f -name "*.py" -exec sed -i "s/from test_project/from $LIBRARY_NAME/g; s/import test_project/import $LIBRARY_NAME/g" {} \;
        success "unit testsが tests/unit/$LIBRARY_NAME/utils/ に移動されました"
    fi

    # Move e2e tests
    if [ -d "tests/e2e/test_project/utils" ]; then
        cp -r tests/e2e/test_project/utils/* "tests/e2e/$LIBRARY_NAME/utils/"
        # Update import paths in test files
        find "tests/e2e/$LIBRARY_NAME/utils" -type f -name "*.py" -exec sed -i "s/from test_project/from $LIBRARY_NAME/g; s/import test_project/import $LIBRARY_NAME/g" {} \;
        success "e2e testsが tests/e2e/$LIBRARY_NAME/utils/ に移動されました"
    fi

    success "utilsテストファイルの移動と更新が完了しました"
else
    info "utilsテストファイルが見つかりません。スキップします。"
fi

# --- Perform File Content Updates ---
CURRENT_YEAR=$(date +%Y)

if [ -f "README.md" ]; then
    info "README.mdを更新しています..."
    sed -i "s/Template Project/$PROJECT_NAME/g" README.md
    sed -i "s/A template Python project with structured logging/$PROJECT_DESCRIPTION/g" README.md
    sed -i "s/MIT/$LICENSE_TYPE/g" README.md
    sed -i "s/template-project/$PROJECT_NAME_SLUG/g" README.md
    success "README.mdが更新されました"
fi

if [ -f "pyproject.toml" ]; then
    info "pyproject.tomlを更新しています..."
    sed -i "s/template-project/$PROJECT_NAME_SLUG/g" pyproject.toml
    sed -i "s/A template Python project with structured logging/$PROJECT_DESCRIPTION/g" pyproject.toml
    sed -i "s/alpha_lib/$LIBRARY_NAME/g" pyproject.toml
    sed -i "s|--cov=src/alpha_lib|--cov=src/$LIBRARY_NAME|g" pyproject.toml
    sed -i "s|include = \[\"src\"\]|include = \[\"src/$LIBRARY_NAME\"\]|g" pyproject.toml
    success "pyproject.tomlが更新されました"
fi

if [ -f "mkdocs.yml" ]; then
    info "mkdocs.ymlを更新しています..."
    sed -i "s/site_name: {{PROJECT_NAME}}/site_name: $PROJECT_NAME/g" mkdocs.yml
    sed -i "s/site_description: {{PROJECT_DESCRIPTION}}/site_description: $PROJECT_DESCRIPTION/g" mkdocs.yml
    success "mkdocs.ymlが更新されました"
fi

# Update LICENSE file
if [ "$LICENSE_TYPE" != "スキップ" ]; then
    info "LICENSEファイルを更新しています..."
    if [ "$LICENSE_TYPE" == "MIT" ]; then
        get_mit_license_text | sed "s/{{YEAR}}/$CURRENT_YEAR/g" | sed "s/{{COPYRIGHT_HOLDER}}/$COPYRIGHT_HOLDER/g" > LICENSE
    elif [ "$LICENSE_TYPE" == "Apache-2.0" ]; then
        get_apache_license_text | sed "s/{{YEAR}}/$CURRENT_YEAR/g" | sed "s/{{COPYRIGHT_HOLDER}}/$COPYRIGHT_HOLDER/g" > LICENSE
    elif [ "$LICENSE_TYPE" == "GPL-3.0" ]; then
        get_gplv3_license_text | sed "s/{{YEAR}}/$CURRENT_YEAR/g" | sed "s/{{COPYRIGHT_HOLDER}}/$COPYRIGHT_HOLDER/g" > LICENSE
    fi
    success "LICENSEファイルが更新されました"
fi

# Check for uv availability
if ! command_exists uv; then
    if [ "$DEVCONTAINER" == "true" ]; then
        error "DevContainer環境でuvが利用できません。postCreateCommand.shが正常に実行されていない可能性があります。"
    else
        info "uvをインストールしています..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        success "uvがインストールされました"
    fi
else
    info "uvは既にインストールされています"
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    info "仮想環境を作成しています..."
    if [ -n "$PYTHON_VERSION" ]; then
        uv venv --python $PYTHON_VERSION || error "仮想環境の作成に失敗しました"
    else
        uv venv || error "仮想環境の作成に失敗しました"
    fi
    success "仮想環境が作成されました"
else
    info "仮想環境は既に存在します"
fi

# Install dependencies
if [ -f "pyproject.toml" ]; then
    info "依存関係をインストールしています..."
    uv sync || error "依存関係のインストールに失敗しました"
    success "依存関係がインストールされました"

    # Install dev dependencies
    info "開発依存関係をインストールしています..."
    uv sync --extra dev || error "開発依存関係のインストールに失敗しました"
    success "開発依存関係がインストールされました"
fi

# Set up pre-commit
if [ -f ".pre-commit-config.yaml" ]; then
    info "pre-commitを設定しています..."
    # Unset core.hooksPath and temporarily isolate from global/system git config
    (export GIT_CONFIG_GLOBAL=/dev/null; export GIT_CONFIG_SYSTEM=/dev/null; git config --unset-all core.hooksPath || true; uv run pre-commit install) || error "pre-commitの設定に失敗しました"
    success "pre-commitが設定されました"
else
    # Try to install pre-commit hooks even if .pre-commit-config.yaml doesn't exist yet
    info "pre-commitフックをインストールしています..."
    if command_exists pre-commit || [ -f ".venv/bin/pre-commit" ]; then
        (export GIT_CONFIG_GLOBAL=/dev/null; export GIT_CONFIG_SYSTEM=/dev/null; git config --unset-all core.hooksPath || true; uv run pre-commit install) || info "pre-commit設定ファイルがありません。後で設定してください。"
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    🎉 セットアップ完了！                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
success "ローカル環境のセットアップが完了しました"
echo ""
info "次のステップ:"
echo "  1. 仮想環境を有効化: source .venv/bin/activate"
echo "  2. コードの実行: uv run python src/$LIBRARY_NAME/main.py"
echo "  3. テストの実行: uv run pytest"
echo "  4. フォーマット: uv run ruff format ."
echo "  5. リント: uv run ruff check ."
echo ""
if [ "$DEVCONTAINER" == "true" ]; then
    info "DevContainer環境で実行中です。VS Codeの統合ターミナルでコマンドを実行してください。"
else
    info "DevContainerを使用する場合は、VS Codeで「Reopen in Container」を選択してください。"
fi
