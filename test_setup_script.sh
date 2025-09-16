#!/bin/bash

# test_setup_script.sh - Tests for setup_github_repo.sh

# --- Configuration ---
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_UNDER_TEST="${BASE_DIR}/setup_github_repo.sh"
MOCK_GH_SCRIPT="${BASE_DIR}/mock_gh.sh" # This is the actual mock_gh.sh script
MOCK_STATE_DIR="/tmp/mock_gh_state"

# --- Test Utilities ---
assert_exit_code() {
  local expected_code="$1"; local actual_code="$2"; local test_name="$3"
  if [[ "$actual_code" -eq "$expected_code" ]]; then
    echo "✓ PASS: ${test_name} - Exit code ${actual_code} as expected."
    return 0
  else
    echo "❌ FAIL: ${test_name} - Expected exit code ${expected_code}, got ${actual_code}."
    return 1
  fi
}

assert_output_contains() {
  local output="$1"; local substring="$2"; local test_name="$3"
  if echo "$output" | grep -qF -- "$substring"; then
    echo "✓ PASS: ${test_name} - Output contains '${substring}'."
    return 0
  else
    echo "❌ FAIL: ${test_name} - Output did not contain '${substring}'. Output was:" >&2
    if ((${#output} > 1000)); then echo "${output:0:1000}..." >&2; else echo "$output" >&2; fi
    return 1
  fi
}

# --- Test State Management ---
setup_mock_state() {
  rm -rf "${MOCK_STATE_DIR}"; mkdir -p "${MOCK_STATE_DIR}"
}
set_mock_not_logged_in() { touch "${MOCK_STATE_DIR}/not_logged_in"; }
set_mock_api_fail_branch_protection() { touch "${MOCK_STATE_DIR}/api_fail_branch_protection"; }

# --- PATH Management & Cleanup ---
PATH_BACKUP_ON_SCRIPT_START="$PATH"
TEMP_MOCK_LINK_DIR="" # To store path to temp dir for symlink

cleanup() {
  export PATH="$PATH_BACKUP_ON_SCRIPT_START"
  if [[ -n "$TEMP_MOCK_LINK_DIR" && -d "$TEMP_MOCK_LINK_DIR" ]]; then
    rm -rf "$TEMP_MOCK_LINK_DIR"
  fi
  rm -rf "${MOCK_STATE_DIR}"
  # echo "Tests finished. Cleaned up." # Optional: for verbosity
}
trap cleanup EXIT INT TERM

# --- Test Cases ---
run_test_gh_not_installed() {
  echo -e "\n--- Test: gh CLI not installed ---"
  setup_mock_state
  local original_path="$PATH"
  export PATH="/usr/bin:/bin:/usr/sbin:/sbin" # Minimal path

  output=$( "$SCRIPT_UNDER_TEST" 2>&1 ); local exit_code=$?
  export PATH="$original_path"

  assert_exit_code 1 "$exit_code" "gh_not_installed_exit_code" && \
  assert_output_contains "$output" "gh CLI not found" "gh_not_installed_message"
  return $?
}

run_test_with_mock_gh() {
  local test_name="$1"
  local setup_mock_function="$2" # Function to set specific mock states (e.g., set_mock_not_logged_in)
  local expected_exit_code="$3"
  local expected_output_substring="$4"
  local output_assertion_message="$5"
  # Optional: input for the script
  local script_input="${6:-}"


  echo -e "\n--- Test: $test_name ---"
  setup_mock_state
  if [[ -n "$setup_mock_function" ]]; then "$setup_mock_function"; fi

  local original_path="$PATH"
  TEMP_MOCK_LINK_DIR=$(mktemp -d) # Create a temp dir for the 'gh' symlink
  ln -s "$MOCK_GH_SCRIPT" "$TEMP_MOCK_LINK_DIR/gh" # Link mock_gh.sh to tempdir/gh
  export PATH="$TEMP_MOCK_LINK_DIR:$original_path" # Prepend temp dir to PATH

  # Debug: Verify gh points to our mock
  # which gh >&2
  # ls -l "$TEMP_MOCK_LINK_DIR/gh" >&2

  local output
  if [[ -n "$script_input" ]]; then
    output=$(echo -e "$script_input" | "$SCRIPT_UNDER_TEST" 2>&1)
  else
    output=$( "$SCRIPT_UNDER_TEST" 2>&1 )
  fi
  local exit_code=$?

  rm -rf "$TEMP_MOCK_LINK_DIR" # Clean up specific temp dir for this test
  TEMP_MOCK_LINK_DIR="" # Reset global
  export PATH="$original_path"  # Restore original PATH for this function's scope

  assert_exit_code "$expected_exit_code" "$exit_code" "${test_name}_exit_code" && \
  assert_output_contains "$output" "$expected_output_substring" "$output_assertion_message"
  return $?
}


# --- Run Actual Tests ---
total_tests=0; passed_tests=0
run_test_wrapper() { # Renamed from run_test to avoid conflict if a test is named run_test
  local test_function_name="$1"; shift # Get the test function
  ((total_tests++))
  if "$test_function_name" "$@"; then # Pass remaining args to the test function
    ((passed_tests++))
  else
    echo "FAIL (details for ${test_function_name} should be above)"
  fi
}

echo "Starting test suite for setup_github_repo.sh..."
# Initial script checks
if [[ ! -x "$MOCK_GH_SCRIPT" ]]; then echo "ERROR: Mock gh script not executable: $MOCK_GH_SCRIPT" >&2; exit 1; fi
if [[ ! -x "$SCRIPT_UNDER_TEST" ]]; then echo "ERROR: Script under test not executable: $SCRIPT_UNDER_TEST" >&2; exit 1; fi

run_test_wrapper run_test_gh_not_installed

run_test_wrapper run_test_with_mock_gh "gh_not_logged_in" \
  "set_mock_not_logged_in" \
  1 \
  "You are not logged into GitHub CLI" \
  "gh_not_logged_in_message"

BRANCH_PROTECTION_INPUT="testowner
testrepo
yes"
run_test_wrapper run_test_with_mock_gh "branch_protection_fails" \
  "set_mock_api_fail_branch_protection" \
  1 \
  "CRITICAL: Failed to protect the 'main' branch" \
  "branch_protection_fail_message" \
  "$BRANCH_PROTECTION_INPUT"


echo -e "\n--- Test Summary ---"
echo "Total tests run: ${total_tests}"; echo "Tests passed:    ${passed_tests}"
if [[ "$passed_tests" -eq "$total_tests" ]]; then
  echo "✓ All tests passed!"; exit 0
else
  echo "❌ Some tests failed."; exit 1
fi
```
