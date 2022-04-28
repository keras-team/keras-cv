#!/bin/bash
base_dir=$(dirname $(dirname $0))
targets="${base_dir}/*.py ${base_dir}/benchmarks/
${base_dir}/examples/
${base_dir}/keras_cv/"

isort --sp "${base_dir}/setup.cfg" --sl --profile=black ${targets}
black ${targets}
