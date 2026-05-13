#!/usr/bin/env bash
#
# Installs the pre-commit hook by creating a symlink from
# .git/hooks/pre-commit to hooks/pre-commit in the repo root.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
GIT_HOOKS_DIR="$(git rev-parse --git-dir)/hooks"
HOOK_SOURCE="${REPO_ROOT}/hooks/pre-commit"
HOOK_TARGET="${GIT_HOOKS_DIR}/pre-commit"

chmod +x "$HOOK_SOURCE"

if [[ -e "$HOOK_TARGET" && ! -L "$HOOK_TARGET" ]]; then
    echo "Warning: ${HOOK_TARGET} already exists and is not a symlink. Backing up to pre-commit.bak."
    mv "$HOOK_TARGET" "${HOOK_TARGET}.bak"
fi

ln -sf "$HOOK_SOURCE" "$HOOK_TARGET"
echo "pre-commit hook installed: ${HOOK_TARGET} -> ${HOOK_SOURCE}"
