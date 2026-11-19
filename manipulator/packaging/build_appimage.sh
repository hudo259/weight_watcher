#!/usr/bin/env bash
set -euo pipefail

# Run from the project root regardless of where the script is called from,
# since all paths below are relative to it. Root = packaging/..
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

# Clean up first
rm -rf build-release AppDir

# Use Ninja when available, otherwise CMake's default (Makefiles, probably)
GENERATOR=()
if command -v ninja >/dev/null; then
    GENERATOR=(-G Ninja)
fi

# Release build installed into a staging AppDir under /usr
cmake -B build-release "${GENERATOR[@]}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
cmake --build build-release
DESTDIR=AppDir cmake --install build-release
ICONDIR=packaging/icon

# linuxdeploy copies remaining .so deps and skips libGL/libEGL by design,
# so the host's GPU driver is used
wget -nc https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
chmod +x linuxdeploy-x86_64.AppImage
./linuxdeploy-x86_64.AppImage --appdir AppDir \
    --desktop-file packaging/weightwatcher.desktop \
    --icon-file "${ICONDIR}/64x64/weightwatcher.png" \
    --icon-file "${ICONDIR}/256x256/weightwatcher.png" \
    --icon-file "${ICONDIR}/512x512/weightwatcher.png" \
    --icon-file "${ICONDIR}/weightwatcher.svg" \
    --output appimage
