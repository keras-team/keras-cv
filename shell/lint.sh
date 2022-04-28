#!/bin/bash
base_dir=$(dirname $(dirname $0))
targets="${base_dir}/*.py ${base_dir}/examples/ ${base_dir}/keras_cv/"

isort --sp "${base_dir}/setup.cfg" --sl -c ${targets}
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
  exit 1
fi
echo "no issues with isort"
flake8 --config "${base_dir}/setup.cfg" ${targets}
if ! [ $? -eq 0 ]
then
  echo "Please fix the code style issue."
  exit 1
fi
echo "no issues with flake8"
black --check ${targets}
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
    exit 1
fi
echo "no issues with black"
for i in $(find ${targets} -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo "Copyright not found in $i"
    exit 1
  fi
done
echo "linting success!"
