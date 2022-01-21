#!/bin/bash
isort --sl -c .
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
  exit 1
fi
echo "no issues with isort"
flake8 .
if ! [ $? -eq 0 ]
then
  echo "Please fix the code style issue."
  exit 1
fi
echo "no issues with flake8"
black --check --line-length 85 .
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
    exit 1
fi
echo "no issues with black"
for i in $(find keras_cv -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo "Copyright not found in $i"
    exit 1
  fi
done
echo "linting success!"
