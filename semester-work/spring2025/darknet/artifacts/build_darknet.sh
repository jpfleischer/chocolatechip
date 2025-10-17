#!/bin/bash
set -euo pipefail

# Use a writable location inside the container
DARKNET_PARENT="${DARKNET_PARENT:-/workspace}"
mkdir -p "$DARKNET_PARENT"
cd "$DARKNET_PARENT" || { echo "Cannot change to $DARKNET_PARENT"; exit 1; }

# Clone Darknet if not already present
if [ ! -d "darknet" ]; then
    git clone https://codeberg.org/jpfleischer/darknet.git
fi

cd darknet || { echo "Cannot change to darknet directory"; exit 1; }

# Ensure we have the latest refs
git remote set-url origin https://codeberg.org/jpfleischer/darknet.git
git fetch --all --tags --prune

# after fetch:
TARGET_REF="${TARGET_REF:-feature/map-reporting-fixes-superclean}"  # default ref

# Try: branch, then tag, then raw commit
if git ls-remote --exit-code --heads origin "$TARGET_REF" >/dev/null 2>&1; then
  git reset --hard
  git checkout -B "$TARGET_REF" "origin/$TARGET_REF"
elif git ls-remote --exit-code --tags origin "$TARGET_REF" >/dev/null 2>&1; then
  git reset --hard
  git checkout -B "tag-$TARGET_REF" "refs/tags/$TARGET_REF"
elif git rev-parse --verify "$TARGET_REF^{commit}" >/dev/null 2>&1; then
  git reset --hard "$TARGET_REF"
  git checkout -B "commit-$TARGET_REF"
else
  echo "ERROR: Ref '$TARGET_REF' not found as branch, tag, or commit."
  echo "Branches:"; git ls-remote --heads origin | awk '{print $2}' | sed 's#refs/heads/##'
  echo "Tags:";     git ls-remote --tags  origin | awk '{print $2}' | sed 's#refs/tags/##'
  exit 1
fi


# Build
mkdir -p build && cd build
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi &>/dev/null; then
    echo "nvidia-smi not found or failed; building CPU-only."
    cmake .. -DDARKNET_TRY_CUDA=OFF -DDARKNET_TRY_ROCM=OFF
else
    echo "nvidia-smi found; building with GPU."
    cmake ..
fi
make -j"$(nproc)"


# Attempt to install the package only if /var/lib/dpkg is writable
if [ -w /var/lib/dpkg ]; then
    dpkg -i /workspace/darknet/build/darknet-*.deb
else
    echo "Skipping package installation: /var/lib/dpkg is read-only."
fi
