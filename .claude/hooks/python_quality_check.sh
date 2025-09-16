#!/bin/bash

# Script to run Python quality checks after file edits
# This is called by Claude Code's PostToolUse hook

# Log that the hook was triggered
echo "[HOOK] Python quality check hook triggered at $(date)" >> /tmp/claude_hook.log

# Change to workspace directory
cd /workspace

# Read the input JSON from stdin
INPUT_JSON=$(cat)

# Also log the input
echo "[HOOK] Input JSON: $INPUT_JSON" >> /tmp/claude_hook.log

# Extract the file path from the JSON input
# Try multiple possible paths in the JSON structure (note: filePath is camelCase)
FILE_PATH=$(echo "$INPUT_JSON" | jq -r '.tool_response.filePath // .tool_input.file_path // empty')

# Log the extracted file path
echo "[HOOK] Extracted file path: $FILE_PATH" >> /tmp/claude_hook.log

# Log the input for debugging
echo "Debug: Hook triggered" >&2
echo "Debug: Extracted file path: $FILE_PATH" >&2

# Check if any Python files were modified recently (fallback approach)
if [ -z "$FILE_PATH" ] || [ "$FILE_PATH" = "null" ]; then
    echo "Debug: No specific file path found, checking for recent Python changes" >&2
    # Find Python files modified in the last minute
    RECENT_PY_FILES=$(find /workspace -name "*.py" -type f -mmin -1 2>/dev/null | head -5)
    if [ -n "$RECENT_PY_FILES" ]; then
        echo "Debug: Found recently modified Python files" >&2
        FILE_PATH="recent_python_files"
    fi
fi

# Check if we should run quality checks
if [[ "$FILE_PATH" == *.py ]] || [ "$FILE_PATH" = "recent_python_files" ]; then
    echo "[HOOK] Running Python quality checks for: $FILE_PATH" >> /tmp/claude_hook.log
    echo "Running Python quality checks..." >&2
    
    cd /workspace
    
    # Initialize error tracking
    ERRORS_FOUND=0
    ERROR_SUMMARY=""
    
    # Run linting
    echo "Running Ruff linting..." >&2
    echo "[HOOK] About to run: uv run nox -s lint" >> /tmp/claude_hook.log
    LINT_OUTPUT=$(uv run nox -s lint 2>&1)
    LINT_EXIT_CODE=$?
    echo "[HOOK] Lint exit code: $LINT_EXIT_CODE" >> /tmp/claude_hook.log
    
    if [ $LINT_EXIT_CODE -ne 0 ]; then
        ERRORS_FOUND=1
        ERROR_SUMMARY="${ERROR_SUMMARY}=== LINTING ERRORS ===\n${LINT_OUTPUT}\n\n"
        echo "[HOOK] Linting failed with output:" >> /tmp/claude_hook.log
        echo "$LINT_OUTPUT" >> /tmp/claude_hook.log
    else
        echo "✓ Linting passed!" >&2
        echo "[HOOK] Linting passed successfully" >> /tmp/claude_hook.log
    fi
    
    # Run format check
    echo "Running code formatting check..." >&2
    echo "[HOOK] About to run: uv run nox -s format_code" >> /tmp/claude_hook.log
    FORMAT_OUTPUT=$(uv run nox -s format_code 2>&1)
    FORMAT_EXIT_CODE=$?
    echo "[HOOK] Format exit code: $FORMAT_EXIT_CODE" >> /tmp/claude_hook.log
    
    if [ $FORMAT_EXIT_CODE -ne 0 ]; then
        ERRORS_FOUND=1
        ERROR_SUMMARY="${ERROR_SUMMARY}=== FORMATTING ERRORS ===\n${FORMAT_OUTPUT}\n\n"
        echo "[HOOK] Formatting failed with output:" >> /tmp/claude_hook.log
        echo "$FORMAT_OUTPUT" >> /tmp/claude_hook.log
    else
        echo "✓ Code formatting check passed!" >&2
        echo "[HOOK] Formatting passed successfully" >> /tmp/claude_hook.log
    fi
    
    # Run import sorting check
    echo "Running import sorting check..." >&2
    echo "[HOOK] About to run: uv run nox -s sort" >> /tmp/claude_hook.log
    SORT_OUTPUT=$(uv run nox -s sort 2>&1)
    SORT_EXIT_CODE=$?
    echo "[HOOK] Sort exit code: $SORT_EXIT_CODE" >> /tmp/claude_hook.log
    
    if [ $SORT_EXIT_CODE -ne 0 ]; then
        ERRORS_FOUND=1
        ERROR_SUMMARY="${ERROR_SUMMARY}=== IMPORT SORTING ERRORS ===\n${SORT_OUTPUT}\n\n"
        echo "[HOOK] Sort failed with output:" >> /tmp/claude_hook.log
        echo "$SORT_OUTPUT" >> /tmp/claude_hook.log
    else
        echo "✓ Import sorting check passed!" >&2
        echo "[HOOK] Sort passed successfully" >> /tmp/claude_hook.log
    fi
    
    # Run type checking
    echo "Running type checking..." >&2
    echo "[HOOK] About to run: uv run nox -s typing" >> /tmp/claude_hook.log
    TYPING_OUTPUT=$(uv run nox -s typing 2>&1)
    TYPING_EXIT_CODE=$?
    echo "[HOOK] Typing exit code: $TYPING_EXIT_CODE" >> /tmp/claude_hook.log
    
    if [ $TYPING_EXIT_CODE -ne 0 ]; then
        ERRORS_FOUND=1
        ERROR_SUMMARY="${ERROR_SUMMARY}=== TYPE CHECKING ERRORS ===\n${TYPING_OUTPUT}\n\n"
        echo "[HOOK] Type checking failed with output:" >> /tmp/claude_hook.log
        echo "$TYPING_OUTPUT" >> /tmp/claude_hook.log
    else
        echo "✓ Type checking passed!" >&2
        echo "[HOOK] Type checking passed successfully" >> /tmp/claude_hook.log
    fi
    
    # Summary output
    if [ $ERRORS_FOUND -eq 1 ]; then
        echo "" >&2
        echo "ERROR: Quality checks failed. Please fix the following issues:" >&2
        echo -e "$ERROR_SUMMARY" >&2
        echo "Note: Quality check errors found but not blocking" >&2
    else
        echo "" >&2
        echo "✓ All quality checks passed!" >&2
    fi
else
    echo "[HOOK] Not a Python file, skipping checks (file: $FILE_PATH)" >> /tmp/claude_hook.log
    echo "Debug: Not a Python file, skipping checks (file: $FILE_PATH)" >&2
fi

# Always exit 0 to avoid blocking
exit 0
