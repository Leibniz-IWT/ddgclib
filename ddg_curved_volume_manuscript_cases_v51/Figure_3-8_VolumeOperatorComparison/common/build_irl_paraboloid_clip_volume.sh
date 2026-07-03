#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

IRL_SRC="$PWD/_external_irl_quadratic_cutting"
EIGEN_SRC="$PWD/_external_eigen"
BUILD_DIR="$PWD/_build_irl"
WRAPPER_SRC="$PWD/irl_paraboloid_clip_volume.cpp"
OUT="$PWD/irl_paraboloid_clip_volume"
JOBS="${JOBS:-2}"
CXX="${CXX:-c++}"
CMAKE_EXE="${CMAKE:-}"

if [ ! -f "$WRAPPER_SRC" ]; then
  echo "Missing wrapper source: $WRAPPER_SRC" >&2
  exit 1
fi
if [ ! -d "$IRL_SRC" ]; then
  echo "Missing IRL source tree: $IRL_SRC" >&2
  exit 1
fi
if [ ! -d "$EIGEN_SRC/Eigen" ]; then
  echo "Missing Eigen headers: $EIGEN_SRC" >&2
  exit 1
fi
if [ -z "$CMAKE_EXE" ]; then
  if command -v cmake >/dev/null 2>&1; then
    CMAKE_EXE="$(command -v cmake)"
  elif [ -x "$PWD/../.tools/cmake/bin/cmake" ]; then
    CMAKE_EXE="$PWD/../.tools/cmake/bin/cmake"
  else
    echo "cmake is required to build the IRL paraboloid helper." >&2
    echo "Install cmake or place a private CMake at ../.tools/cmake/bin/cmake." >&2
    exit 1
  fi
fi

"$CMAKE_EXE" -S "$IRL_SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEIGEN_PATH="$EIGEN_SRC" \
  -DUSE_ABSL=OFF \
  -DBUILD_TESTING=OFF \
  -DPARABOLOID_TESTING=OFF \
  -DCYLINDER_TESTING=OFF \
  -DIRL_BUILD_FORTRAN=OFF

"$CMAKE_EXE" --build "$BUILD_DIR" --target irl --parallel "$JOBS"

"$CXX" -std=c++17 -O3 -DNDEBUG -D IRL_NO_ABSL \
  -I"$IRL_SRC" -I"$EIGEN_SRC" \
  "$WRAPPER_SRC" "$BUILD_DIR/libirl.a" \
  -lquadmath \
  -o "$OUT"

chmod +x "$OUT"
echo "Built $OUT"
