#!/bin/bash

# Script to run comprehensive Python quality checks before stopping
# This is called by Claude Code's Stop and SubagentStop hooks

echo "Running final quality checks before stopping..."

# Flag to track if any checks fail
CHECKS_FAILED=false
ERROR_MESSAGES=""

# Run tests
echo "Running tests..."
TEST_OUTPUT=$(uv run nox -s test 2>&1)
TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -ne 0 ]; then
    CHECKS_FAILED=true
    ERROR_MESSAGES="${ERROR_MESSAGES}TEST FAILURES:\n${TEST_OUTPUT}\n\n"
fi

# Run linting
echo "Running linting..."
LINT_OUTPUT=$(uv run nox -s lint 2>&1)
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
    CHECKS_FAILED=true
    ERROR_MESSAGES="${ERROR_MESSAGES}LINTING ERRORS:\n${LINT_OUTPUT}\n\n"
fi

# Run type checking
echo "Running type checking..."
TYPING_OUTPUT=$(uv run nox -s typing 2>&1)
TYPING_EXIT_CODE=$?
if [ $TYPING_EXIT_CODE -ne 0 ]; then
    CHECKS_FAILED=true
    ERROR_MESSAGES="${ERROR_MESSAGES}TYPE CHECKING ERRORS:\n${TYPING_OUTPUT}\n\n"
fi

# If any checks failed, output error and exit with code 2
if [ "$CHECKS_FAILED" = true ]; then
    echo "ERROR: Quality checks failed. Please fix the following issues before stopping:" >&2
    echo -e "$ERROR_MESSAGES" >&2
    exit 2
fi

echo "All quality checks passed!"
exit 0
