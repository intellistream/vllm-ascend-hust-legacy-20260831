#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <package> [package ...]" >&2
  exit 2
fi

missing_packages=()
for package_name in "$@"; do
  if ! command -v "$package_name" >/dev/null 2>&1; then
    missing_packages+=("$package_name")
  fi
done

if [[ "${#missing_packages[@]}" -eq 0 ]]; then
  echo "[OK] CI system packages are already available: $*"
  exit 0
fi

run_as_root=()
if [[ "$(id -u)" -ne 0 ]]; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "[ERROR] Missing CI system packages: ${missing_packages[*]}" >&2
    echo "[ERROR] sudo is unavailable; install these packages in the runner image or host bootstrap." >&2
    exit 1
  fi
  run_as_root=(sudo -n)
fi

echo "[INFO] Installing missing CI system packages: ${missing_packages[*]}"

if command -v apt-get >/dev/null 2>&1; then
  "${run_as_root[@]}" apt-get update -y
  "${run_as_root[@]}" apt-get install -y --no-install-recommends "${missing_packages[@]}"
elif command -v dnf >/dev/null 2>&1; then
  "${run_as_root[@]}" dnf install -y "${missing_packages[@]}"
elif command -v yum >/dev/null 2>&1; then
  "${run_as_root[@]}" yum install -y "${missing_packages[@]}"
elif command -v apk >/dev/null 2>&1; then
  "${run_as_root[@]}" apk add --no-cache "${missing_packages[@]}"
else
  echo "[ERROR] Unsupported package manager. Missing CI system packages: ${missing_packages[*]}" >&2
  exit 1
fi

for package_name in "${missing_packages[@]}"; do
  if ! command -v "$package_name" >/dev/null 2>&1; then
    echo "[ERROR] Package installation completed but command is still unavailable: $package_name" >&2
    exit 1
  fi
done

echo "[OK] Installed CI system packages: ${missing_packages[*]}"
