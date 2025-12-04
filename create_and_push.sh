#!/usr/bin/env bash
set -e

REPO_OWNER="KOdali-abhi"
REPO_NAME="noise-learn"

echo "Ensure you're in the project root with README.md, pyproject.toml, src/, examples/ ..."
read -p "Press Enter to continue or Ctrl-C to cancel"

# init repo if needed
if [ ! -d .git ]; then
  git init
fi

git add .
git commit -m "Initial iladok / noise-learn scaffold" || echo "Nothing to commit"

# Try GH CLI route first
if command -v gh >/dev/null 2>&1; then
  echo "Using gh CLI to create and push repository..."
  gh auth status >/dev/null 2>&1 || { echo "Please run: gh auth login"; exit 1; }
  gh repo create ${REPO_OWNER}/${REPO_NAME} --public --source=. --remote=origin --push --confirm
  echo "Repository created and pushed via gh CLI."
  exit 0
fi

# Fallback: interactive instruction for manual creation
echo "gh CLI not found. Please create the repository at https://github.com/new (Owner=${REPO_OWNER}, Name=${REPO_NAME}), then run:"
echo "  git remote add origin git@github.com:${REPO_OWNER}/${REPO_NAME}.git"
echo "  git branch -M main"
echo "  git push -u origin main"
exit 0