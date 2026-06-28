#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Load rbenv so the correct Ruby version is used ──────────────────────────
export PATH="$HOME/.rbenv/bin:$PATH"
eval "$(rbenv init -)"

# ── Build ────────────────────────────────────────────────────────────────────
echo "==> Building site..."
JEKYLL_ENV=production bundle exec jekyll build --destination docs

# ── Commit & push ───────────────────────────────────────────────────────────
echo "==> Staging changes..."
git add -A

if git diff --cached --quiet; then
  echo "Nothing to commit — site is already up to date."
  exit 0
fi

COMMIT_MSG="${1:-"build: update site $(date '+%Y-%m-%d %H:%M')"}"
git commit -m "$COMMIT_MSG"

echo "==> Pushing to GitHub..."
git push

echo "==> Done. Your site will be live at https://chuan2019.github.io in ~1 minute."
