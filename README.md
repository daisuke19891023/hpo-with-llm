# HPO with LLM

大規模言語モデル (LLM) を活用したハイパーパラメータ最適化 (HPO) ワークフローを、CLI と REST API の両方で扱える Python プロジェクトです。HPO の実行、試行履歴のロギング、LLM を用いた振り返り (Reflection) を共通の基盤上で行えるように設計されています。

## 主な機能

- **統合された HPO オーケストレーション**: `HPOOrchestrator` がサーチバックエンドと試行実行関数を仲介し、試行の実行・結果集計を管理します。デフォルトではインメモリ探索バックエンドとサンプルの試行実行関数が利用されます。
- **柔軟なハイパーパラメータ定義**: YAML で記述した検索空間を読み込み、環境変数・CLI 引数・設定ファイル (YAML/テキスト) への値適用まで自動生成できます。
- **LLM を用いた振り返り**: ベースラインの統計分析に加え、OpenAI/Azure OpenAI などの LLM を使った振り返りモードを備えています。
- **複数インターフェース**: Typer ベースの CLI に加えて FastAPI 実装の REST API を提供し、同じ HPO 基盤を別の UI から呼び出せます。
- **構造化ロギングと設定**: `.env` や環境変数から設定を読み込み、Structlog/OTel を利用したロギングを構成できます。

## ディレクトリ構造

```
├── src/clean_interfaces/
│   ├── app.py                # アプリケーション組み立てと HPO ヘルパー
│   ├── main.py               # CLI エントリーポイント
│   ├── interfaces/           # CLI・REST API などのインターフェース
│   ├── hpo/                  # オーケストレーション・バックエンド・設定
│   ├── llm/                  # LLM クライアントと設定
│   ├── evaluation/           # 評価ユーティリティ (例: ゴールデンデータ)
│   └── utils/                # 設定・ロギングなどの共通ユーティリティ
├── docs/                     # MkDocs ベースのドキュメント
├── tests/                    # 単体・統合テスト
├── env.example               # `.env` のひな形
├── noxfile.py                # 開発タスク定義
├── pyproject.toml            # パッケージ設定
└── README.md                 # 本ファイル
```

## セットアップ

### 前提条件

- Python 3.13 以上
- [uv](https://github.com/astral-sh/uv) (推奨) ※ もしくは `pip`
- Git (リポジトリの取得用)

### リポジトリの取得

```bash
git clone <repository-url>
cd hpo-with-llm
```

### 依存関係のインストール

uv を利用する場合:

```bash
uv sync
```

pip を利用する場合 (開発ツール込み):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 環境変数と設定

- `.env` を使う場合は `cp env.example .env` でコピーし、必要に応じて値を編集します。
- 主要な環境変数 (一例):

| 変数 | 説明 | デフォルト |
| --- | --- | --- |
| `INTERFACE_TYPE` | 起動するインターフェース (`cli` / `restapi`) | `cli` |
| `LOG_LEVEL` | ログレベル (`DEBUG` / `INFO` など) | `INFO` |
| `LOG_FORMAT` | ログ形式 (`json` / `console` / `plain`) | `json` |
| `LOG_FILE_PATH` | ファイル出力パス (任意) | 未設定 |
| `OTEL_LOGS_EXPORT_MODE` | OpenTelemetry 出力 (`file` / `otlp` / `both`) | `file` |
| `OTEL_ENDPOINT` | OTLP エンドポイント | `http://localhost:4317` |
| `LLM_PROVIDER` | 利用する LLM プロバイダ (`openai` / `azure_openai`) | `openai` |
| `LLM_MODEL` | 既定で利用するモデル名 | `gpt-4o-mini` |
| `OPENAI_API_KEY` 等 | 各プロバイダの API キー | 未設定 |

`.env` 以外に CLI の `--dotenv` オプションで別ファイルを読み込むこともできます。

## 使い方

### CLI インターフェース

まずはヘルプを確認します:

```bash
uv run python -m clean_interfaces.main -- --help
```

主なサブコマンド:

- ウェルカムメッセージ
  ```bash
  uv run python -m clean_interfaces.main welcome
  ```
- HPO の実行 (デフォルトの探索空間)
  ```bash
  uv run python -m clean_interfaces.main run-hpo \
    --task "Improve retrieval strategy" \
    --max-trials 5 \
    --direction maximize
  ```
- 振り返り付き HPO (LLM モードを利用)
  ```bash
  uv run python -m clean_interfaces.main reflect-hpo \
    --task "Tune RAG pipeline" \
    --mode llm \
    --max-trials 3
  ```

#### カスタム検索空間を使う
`clean_interfaces.hpo.configuration.load_tuning_config` は YAML 形式の検索空間を読み込みます。下記のようなファイルを用意し、`--search-space-config` で指定できます。

```yaml
parameters:
  - name: temperature
    type: float
    lower: 0.0
    upper: 1.0
    description: Sampling temperature
    location:
      type: environment
      variable: LLM_TEMPERATURE
  - name: max_output_tokens
    type: int
    lower: 128
    upper: 1024
    step: 64
    location:
      type: cli_argument
      flag: --max-output-tokens
      value_template: "{value}"
  - name: retrieval_strategy
    type: categorical
    choices: [keyword, vector, hybrid]
```

```bash
uv run python -m clean_interfaces.main run-hpo \
  --task "Custom search" \
  --search-space-config ./tuning.yaml
```

### REST API インターフェース

REST API を起動する場合はインターフェース種別を切り替えます。

```bash
INTERFACE_TYPE=restapi uv run python -m clean_interfaces.main
```

既定では FastAPI サーバーが `http://127.0.0.1:8000` で立ち上がります。主なエンドポイントは以下の通りです。

- `GET /health` ヘルスチェック
- `GET /api/v1/welcome` ウェルカムメッセージ
- `POST /api/v1/hpo/run` HPO 実行 (JSON で `task`, `search_space`, `config` を指定)
- `GET /api/v1/swagger-ui` Swagger UI (拡張ドキュメント)

### Python から利用する

コード内から直接呼び出す場合は `clean_interfaces.app` のヘルパーを利用できます。

```python
from clean_interfaces.app import run_hpo_experiment, run_hpo_with_reflection
from clean_interfaces.hpo.executors import default_trial_executor

result = run_hpo_experiment(request, trial_executor=default_trial_executor)
result, reflection = run_hpo_with_reflection(
    request,
    trial_executor=default_trial_executor,
)
```

## 開発環境の整備

```bash
# 開発用依存関係の追加インストール
uv sync --extra dev

# pre-commit フックの設定
uv run pre-commit install
```

### Nox コマンド

| コマンド | 説明 |
| --- | --- |
| `nox -s lint` | 静的解析 (Ruff) |
| `nox -s format_code` | フォーマッタ実行 |
| `nox -s typing` | 型チェック (Pyright) |
| `nox -s test` | テスト一式 |
| `nox -s security` | セキュリティチェック |
| `nox -s docs` | ドキュメントビルド |
| `nox -s ci` | CI 想定の全チェック |

## ドキュメント

`docs/` 配下には詳細な設定やガイドが整理されています。必要に応じて `nox -s docs` で静的サイトをビルドし、ブラウザで参照できます。

## ライセンス

本プロジェクトは MIT License の下で提供されています。
