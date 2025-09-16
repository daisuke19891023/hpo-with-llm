#!/bin/bash

# setup_github_repo.sh
# This script helps configure a GitHub repository for OSS development best practices.

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              GitHub Repository Setup Script                 ║"
echo "║           For Python + Node.js Projects                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "This script will guide you through setting up your GitHub repository with recommended configurations:"
echo "  🔒 Protect the main branch (require PR reviews, prevent force pushes, etc.)"
echo "  📝 Enable issue and pull request templates guidance"
echo "  🛡️ Enable vulnerability alerts"
echo "  🤖 Enable automated security fixes (Dependabot)"
echo "  ✅ Configure branch protection to require CI checks and up-to-date branches"
echo ""
echo "Prerequisites:"
echo "  • GitHub CLI ('gh') must be installed and authenticated"
echo "  • You must have admin rights to the repository you want to configure"
echo "  • CI workflow (.github/workflows/ci.yml) should already exist"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Function to check if gh CLI is installed and user is logged in
check_gh_installed_and_auth() {
  if ! command -v gh &> /dev/null
  then
    echo "❌ ERROR: gh CLI not found."
    echo "   Please install gh CLI from https://cli.github.com/ and ensure it's in your PATH."
    exit 1
  fi
  echo "✅ gh CLI is installed."

  # Check gh auth status
  if ! gh auth status &> /dev/null; then
    echo "❌ ERROR: You are not logged into GitHub CLI."
    echo "   Please run 'gh auth login' to authenticate, then try running this script again."
    exit 1
  fi
  echo "✅ Authenticated with GitHub CLI."
}

# Function to get repository information
get_repo_info() {
  echo ""
  echo "Please enter the details of the GitHub repository you want to configure:"
  read -p "📦 Enter the repository owner (e.g., your-username): " REPO_OWNER
  while [[ -z "$REPO_OWNER" ]]; do
    echo "❌ Repository owner cannot be empty."
    read -p "📦 Enter the repository owner (e.g., your-username): " REPO_OWNER
  done

  read -p "📁 Enter the repository name (e.g., your-repo-name): " REPO_NAME
  while [[ -z "$REPO_NAME" ]]; do
    echo "❌ Repository name cannot be empty."
    read -p "📁 Enter the repository name (e.g., your-repo-name): " REPO_NAME
  done
  echo ""
}

# Function to detect project type and suggest CI checks
detect_project_type() {
  PROJECT_TYPE=""
  CI_CHECKS='[]'

  if [[ -f "pyproject.toml" && -f "package.json" ]]; then
    PROJECT_TYPE="Python + Node.js"
    CI_CHECKS='["test-python", "test-node", "lint"]'
  elif [[ -f "pyproject.toml" ]]; then
    PROJECT_TYPE="Python"
    CI_CHECKS='["test", "lint"]'
  elif [[ -f "package.json" ]]; then
    PROJECT_TYPE="Node.js"
    CI_CHECKS='["test", "lint"]'
  else
    PROJECT_TYPE="Generic"
    CI_CHECKS='["test"]'
  fi

  echo "🔍 Detected project type: $PROJECT_TYPE"
}

# Function to protect the main branch with CI requirements
protect_main_branch() {
  echo "⏳ Attempting to protect the main branch for ${REPO_OWNER}/${REPO_NAME}..."

  # Check if CI workflow exists
  if [[ ! -f ".github/workflows/ci.yml" ]]; then
    echo "⚠️ WARNING: CI workflow file (.github/workflows/ci.yml) not found."
    echo "   Branch protection will be set without specific CI status checks."
    echo "   You can manually configure status checks later in GitHub repository settings."
    # Set status_checks to empty array if no CI workflow
    CONTEXTS='[]'
  else
    echo "✅ Found existing CI workflow file: .github/workflows/ci.yml"
    # Use detected project-specific checks
    CONTEXTS=$CI_CHECKS
  fi

  # Get GitHub token from gh CLI
  GITHUB_TOKEN=$(gh auth token)

  # Create JSON payload (minimal required fields)
  JSON_PAYLOAD=$(cat <<EOF
{
  "required_status_checks": {
    "strict": true,
    "contexts": ${CONTEXTS}
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "restrictions": null
}
EOF
)

  # Set branch protection using curl
  RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X PUT \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -d "${JSON_PAYLOAD}" \
    "https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/branches/main/protection")

  # Extract HTTP status code
  HTTP_CODE=$(echo "${RESPONSE}" | tail -n1)
  RESPONSE_BODY=$(echo "${RESPONSE}" | head -n -1)

  if [ "${HTTP_CODE}" -eq 200 ]; then
    echo "✅ Successfully protected the main branch."
    echo "   📋 Branch Protection Rules Applied:"
    echo "   • Enforce admin bypass prevention: On"
    echo "   • Prevent force pushes: On"
    echo "   • Prevent deletions: On"
    echo "   • Require pull request reviews: 1 approval required"
    echo "   • Require branches to be up to date before merging: On"
    if [[ -f ".github/workflows/ci.yml" ]]; then
      echo "   • Require status checks: ${PROJECT_TYPE} CI jobs required"
    fi
    echo ""
  else
    echo "❌ CRITICAL: Failed to protect the 'main' branch for ${REPO_OWNER}/${REPO_NAME}."
    echo "   HTTP Status Code: ${HTTP_CODE}"
    echo "   Response: ${RESPONSE_BODY}"
    echo "   This is a critical step. Further repository setup might be incomplete or insecure."
    echo "   Please check the repository path, your permissions, and gh CLI authentication."
    exit 1
  fi
}

# Function to enable issue templates
enable_issue_templates() {
  echo "⏳ Configuring issue templates guidance for ${REPO_OWNER}/${REPO_NAME}..."
  gh api \
    --method PATCH \
    -H "Accept: application/vnd.github.v3+json" \
    "/repos/${REPO_OWNER}/${REPO_NAME}" \
    -f has_issues=true \
    --silent

  if [ $? -eq 0 ]; then
    echo "✅ Issue tracking has been confirmed/enabled."
    echo "   💡 Reminder: Create detailed issue templates in '.github/ISSUE_TEMPLATE'."
  else
    echo "⚠️ WARNING: Failed to update repository settings for issue templates guidance for ${REPO_OWNER}/${REPO_NAME}."
    echo "   Please check your permissions. This is not a critical failure; script will continue."
  fi
}

# Function to enable pull request templates
enable_pr_templates() {
  echo "ℹ️ Configuring pull request templates guidance for ${REPO_OWNER}/${REPO_NAME}..."
  echo "   💡 Pull request templates are encouraged for consistency."
  echo "   💡 Reminder: Create templates in '.github/PULL_REQUEST_TEMPLATE/' or as '.github/PULL_REQUEST_TEMPLATE.md'."
}

# Function to enable vulnerability alerts
enable_vulnerability_alerts() {
  echo "⏳ Enabling vulnerability alerts for ${REPO_OWNER}/${REPO_NAME}..."
  gh api \
    --method PUT \
    -H "Accept: application/vnd.github.v3+json" \
    "/repos/${REPO_OWNER}/${REPO_NAME}/vulnerability-alerts" \
    --silent

  if [ $? -eq 0 ]; then
    echo "✅ Successfully enabled vulnerability alerts for ${REPO_OWNER}/${REPO_NAME}."
  else
    echo "⚠️ WARNING: Failed to enable vulnerability alerts for ${REPO_OWNER}/${REPO_NAME}."
    echo "   This feature might require admin rights or a specific GitHub plan."
    echo "   Please check your permissions. Script will continue."
  fi
}

# Function to enable automated security fixes (Dependabot)
enable_automated_security_fixes() {
  echo "⏳ Enabling automated security fixes (Dependabot) for ${REPO_OWNER}/${REPO_NAME}..."
  gh api \
    --method PUT \
    -H "Accept: application/vnd.github.v3+json" \
    "/repos/${REPO_OWNER}/${REPO_NAME}/automated-security-fixes" \
    --silent

  if [ $? -eq 0 ]; then
    echo "✅ Successfully enabled automated security fixes (Dependabot) for ${REPO_OWNER}/${REPO_NAME}."
    echo "   💡 Consider adding a .github/dependabot.yml file for custom configuration."
  else
    echo "⚠️ WARNING: Failed to enable automated security fixes (Dependabot) for ${REPO_OWNER}/${REPO_NAME}."
    echo "   This feature might require admin rights or a specific GitHub plan (e.g., GitHub Advanced Security)."
    echo "   Please check your permissions. Script will continue."
  fi
}

# --- Main script execution ---
check_gh_installed_and_auth
get_repo_info
detect_project_type

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "You are about to apply settings to the following repository:"
echo "📦 Owner: ${REPO_OWNER}"
echo "📁 Name:  ${REPO_NAME}"
echo "🔍 Type:  ${PROJECT_TYPE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "Do you want to proceed with applying these settings? (yes/no): " CONFIRMATION

if [[ "${CONFIRMATION,,}" != "yes" ]]; then
  echo "❌ Configuration aborted by the user."
  exit 0
fi

echo ""
echo "🚀 Proceeding with repository configuration..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

protect_main_branch
enable_issue_templates
enable_pr_templates
enable_vulnerability_alerts
enable_automated_security_fixes

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 GitHub Repository Setup Script Finished!"
echo "All configured settings have been applied (or attempted) for ${REPO_OWNER}/${REPO_NAME}."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Next Steps / Manual Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "• Consider creating detailed issue templates in the '.github/ISSUE_TEMPLATE' directory"
echo "• Consider creating pull request templates in the '.github/PULL_REQUEST_TEMPLATE' directory"
echo "• Branch protection is now active - all PRs must be up-to-date with main branch and have required reviews"
echo "• If you have an existing CI workflow (.github/workflows/ci.yml), it's configured as a required status check"
echo "• Review and adjust your CI workflow as needed for your ${PROJECT_TYPE} project"
echo "• For Python projects: ensure pytest, ruff, and other tools are included in CI"
echo "• For Node.js projects: ensure npm test, eslint, and other tools are included in CI"
echo "• Explore other repository settings that might be relevant to your project (e.g., Pages, webhooks, environments)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit 0
