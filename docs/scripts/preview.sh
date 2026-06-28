#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.rbenv/bin:$PATH"
eval "$(rbenv init -)"

echo "==> Starting local preview at http://localhost:4000 ..."
bundle exec jekyll serve --livereload --open-url
