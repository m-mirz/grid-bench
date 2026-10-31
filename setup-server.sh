#!/usr/bin/env bash
# Quick start on a rented server (Debian/Ubuntu, a sudo user): tmux, Docker,
# Claude Code, grid-bench (with its dataset submodules) and gridoxide cloned
# side by side over HTTPS, then a tmux session "bench" in grid-bench for
# docker/build.sh and docker/run_benchmark.sh, so a dropped SSH connection
# loses nothing (reattach: tmux attach -t bench).
# No uv or native Python: everything runs in the images (setup.sh is the native path).
# The gridoxide clone is for unreleased work: the build uses the pinned release
# unless GRIDOXIDE_REPO=../gridoxide is set (docker/build.sh).
# Usage: curl -fsSL https://raw.githubusercontent.com/m-mirz/grid-bench/main/setup-server.sh | bash
#        (clones into the current directory)
set -euo pipefail
sudo apt-get update
sudo apt-get install -y git tmux
command -v docker >/dev/null || sudo apt-get install -y docker.io docker-compose-v2
sudo adduser "$(id -un)" docker
command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash
grep -qs 'HOME/.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
[ -d grid-bench ] || git clone --recursive https://github.com/m-mirz/grid-bench.git
[ -d gridoxide ] || git clone --recursive https://github.com/m-mirz/gridoxide.git
cd grid-bench
# Group membership applies to new logins only; sg gives it to the tmux server,
# and so to every shell in the session, without logging in again.
exec sg docker -c "tmux new -s bench" </dev/tty
