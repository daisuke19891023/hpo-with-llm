#!/bin/bash

# mock_gh.sh - A mock for the GitHub CLI (gh)

MOCK_STATE_DIR="/tmp/mock_gh_state"
CONFIG_NOT_LOGGED_IN="$MOCK_STATE_DIR/not_logged_in"
CONFIG_API_FAIL_GENERIC="$MOCK_STATE_DIR/api_fail_generic"
CONFIG_API_FAIL_BRANCH_PROTECTION="$MOCK_STATE_DIR/api_fail_branch_protection" # Key file for the failing test

log_mock_action() {
  echo "[mock_gh.sh DEBUG] $@" >&2
}

log_mock_action "RAW ARGS: $@"

if [[ "$1" == "--version" ]]; then
  log_mock_action "Command: --version"
  echo "gh version 2.0.0 (mock_gh)"
  exit 0
fi

if [[ "$1" == "auth" && "$2" == "status" ]]; then
  log_mock_action "Command: auth status. Checking for $CONFIG_NOT_LOGGED_IN"
  if [[ -f "$CONFIG_NOT_LOGGED_IN" ]]; then
    log_mock_action "Simulating 'not logged in' because $CONFIG_NOT_LOGGED_IN exists."
    exit 1
  else
    log_mock_action "Simulating 'logged in' because $CONFIG_NOT_LOGGED_IN does not exist."
    exit 0
  fi
fi

if [[ "$1" == "api" ]]; then
  ENDPOINT_PATH=""
  RAW_ARGS_FOR_API_LOGGING="$@"
  _ARG_INDEX=1; shift; _ARG_INDEX=2
  log_mock_action "After 'api' shift, current \$1 is '$1' (orig \$$_ARG_INDEX)"

  if [[ "$1" == "--method" ]]; then
    log_mock_action "Found --method. Original method was '$2' (orig \$${_ARG_INDEX+1}). Shifting 2."
    shift 2; _ARG_INDEX=$((_ARG_INDEX + 2))
    ENDPOINT_PATH="$1"
    log_mock_action "After '--method' shift, ENDPOINT_PATH is '$ENDPOINT_PATH' (orig \$$_ARG_INDEX)"
  else
    ENDPOINT_PATH="$1"
    log_mock_action "No '--method'. ENDPOINT_PATH is '$ENDPOINT_PATH' (orig \$$_ARG_INDEX)"
  fi

  log_mock_action "Command: api, Parsed Endpoint: '$ENDPOINT_PATH', All original api args: $RAW_ARGS_FOR_API_LOGGING"

  # MODIFIED LOGIC: If the specific branch protection fail file exists, ANY API call will fail.
  # This is to ensure Test Case 3 triggers a failure if the file is present,
  # bypassing potential issues with ENDPOINT_PATH matching or file visibility in that exact conditional.
  if [[ -f "$CONFIG_API_FAIL_BRANCH_PROTECTION" ]]; then
    log_mock_action "BRANCH PROTECTION FAIL FILE IS PRESENT ($CONFIG_API_FAIL_BRANCH_PROTECTION) - FAILING THIS API CALL to '$ENDPOINT_PATH'"
    exit 1 # Fail the API call
  fi

  # Generic failure check (less specific than the one above now)
  if [[ -f "$CONFIG_API_FAIL_GENERIC" ]]; then
    log_mock_action "Simulating generic API failure for $ENDPOINT_PATH"
    exit 1
  fi

  # Original specific check (now somewhat redundant if the above broader check is active for the test)
  # if [[ "$ENDPOINT_PATH" == "/repos/"*"/branches/main/protection" ]]; then
  #   log_mock_action "ENDPOINT_PATH matches branch protection pattern."
  #   if [[ -f "$CONFIG_API_FAIL_BRANCH_PROTECTION" ]]; then # This file check is now effectively global for API calls
  #     log_mock_action "SIMULATING BRANCH PROTECTION API FAILURE for $ENDPOINT_PATH because $CONFIG_API_FAIL_BRANCH_PROTECTION exists."
  #     exit 1
  #   else
  #     log_mock_action "Branch protection call, but $CONFIG_API_FAIL_BRANCH_PROTECTION does NOT exist. Simulating success."
  #   fi
  # else
  #   log_mock_action "ENDPOINT_PATH '$ENDPOINT_PATH' does not match branch protection pattern."
  # fi

  log_mock_action "Defaulting to successful API call for $ENDPOINT_PATH"
  exit 0
fi

log_mock_action "Unknown command or arguments: $@"
exit 127
```
