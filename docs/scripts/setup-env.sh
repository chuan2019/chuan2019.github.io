#!/usr/bin/env bash
#
# One-time setup of the Ruby build environment for this Jekyll site.
#
# Installs (idempotently):
#   1. System build dependencies (via apt — requires sudo)
#   2. rbenv + ruby-build   -> ~/.rbenv
#   3. Ruby 3.3.8           (the version pinned in .ruby-version)
#   4. Bundler + project gems into ./vendor/bundle (the isolated gem set)
#
# Safe to re-run: every step is skipped if already satisfied.
#
# Usage:   bash scripts/setup-env.sh
# Then:    scripts/deploy.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUBY_VERSION="$(cat .ruby-version)"   # 3.3.8

echo "==> [1/4] Installing system build dependencies (sudo may prompt for your password)..."
sudo apt-get update
sudo apt-get install -y \
  git curl build-essential autoconf patch rustc \
  libssl-dev libyaml-dev libreadline-dev zlib1g-dev \
  libgmp-dev libncurses-dev libffi-dev libgdbm-dev libdb-dev uuid-dev

echo "==> [2/4] Installing rbenv + ruby-build..."
if [ ! -d "$HOME/.rbenv" ]; then
  git clone https://github.com/rbenv/rbenv.git "$HOME/.rbenv"
fi
if [ ! -d "$HOME/.rbenv/plugins/ruby-build" ]; then
  git clone https://github.com/rbenv/ruby-build.git "$HOME/.rbenv/plugins/ruby-build"
else
  git -C "$HOME/.rbenv/plugins/ruby-build" pull --ff-only || true
fi

# Make rbenv available in this script...
export PATH="$HOME/.rbenv/bin:$PATH"
eval "$(rbenv init - bash)"

# ...and in future interactive shells (so `deploy.sh` / `preview.sh` work later).
if ! grep -q 'rbenv init' "$HOME/.bashrc" 2>/dev/null; then
  {
    echo ''
    echo '# rbenv (Ruby version manager)'
    echo 'export PATH="$HOME/.rbenv/bin:$PATH"'
    echo 'eval "$(rbenv init - bash)"'
  } >> "$HOME/.bashrc"
  echo "    Added rbenv init to ~/.bashrc"
fi

echo "==> [3/4] Installing Ruby ${RUBY_VERSION} (compiles from source; this takes a few minutes)..."
rbenv install -s "$RUBY_VERSION"
rbenv global "$RUBY_VERSION"
rbenv rehash
echo "    Ruby: $(ruby -v)"

echo "==> [4/4] Installing project gems into ./vendor/bundle ..."
gem install bundler -v "$(tail -1 Gemfile.lock | tr -d ' ')" --conservative
bundle config set --local path 'vendor/bundle'
bundle install

echo ""
echo "==> Done. Environment ready."
echo "    Next: run  scripts/deploy.sh   to build docs/ and publish your post."
