#!/usr/bin/env bash
# Copyright 2022 The KerasCV Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Builds a wheel of KerasCV for Pip. Requires Bazel.
# Adapted from https://github.com/tensorflow/addons/blob/master/build_deps/build_pip_pkg.sh

set -e
set -x

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
function is_windows() {
  if [[ "${PLATFORM}" =~ (cygwin|mingw32|mingw64|msys)_nt* ]]; then
    true
  else
    false
  fi
}

if is_windows; then
  PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.exe.runfiles/__main__/"
else
  PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.runfiles/__main__/"
fi

function main() {
  while [[ ! -z "${1}" ]]; do
    if [[ ${1} == "make" ]]; then
      echo "Using Makefile to build pip package."
      PIP_FILE_PREFIX=""
    else
      DEST=${1}
    fi
    shift
  done

  if [[ -z ${DEST} ]]; then
    echo "No destination dir provided"
    exit 1
  fi

  # Create the directory, then do dirname on a non-existent file inside it to
  # give us an absolute paths with tilde characters resolved to the destination
  # directory.
  mkdir -p ${DEST}
  if [[ ${PLATFORM} == "darwin" ]]; then
    DEST=$(pwd -P)/${DEST}
  else
    DEST=$(readlink -f "${DEST}")
  fi
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy KerasCV Custom op files"

  cp ${PIP_FILE_PREFIX}setup.cfg "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}setup.py "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}MANIFEST.in "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}README.md "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}LICENSE "${TMPDIR}"

  if is_windows; then
   from=$(cygpath -w ${PIP_FILE_PREFIX}keras_cv)
   to=$(cygpath -w "${TMPDIR}"/keras_cv)
   start robocopy //S "${from}" "${to}" //xf *_test.py
   sleep 5
  else
   rsync -avm -L --exclude='*_test.py' ${PIP_FILE_PREFIX}keras_cv "${TMPDIR}"
  fi

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  python setup.py bdist_wheel > /dev/null

  cp dist/*.whl "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
