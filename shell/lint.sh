#!/bin/bash
# Usage: # lint.sh can be used without arguments to lint the entire project:
#
# ./lint.sh
#
# or with arguments to lint a subset of files
#
# ./lint.sh examples/*

files="."
if [ $# -ne 0  ]
  then
    files=$@
fi

isort -c $files
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
  isort --version
  black --version
  exit 1
fi
[ $# -eq 0  ] && echo "no issues with isort"

# Allow --max-line-length=200 to support long links in docstrings
flake8 --max-line-length=200 $files
if ! [ $? -eq 0 ]
then
  echo "Please fix the code style issue."
  exit 1
fi
[ $# -eq 0 ] && echo "no issues with flake8"
black --check $files
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
    exit 1
fi
[ $# -eq 0  ] && echo "no issues with black"
for i in $(find keras_cv -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo "Copyright not found in $i"
    exit 1
  fi
done
echo "linting success!"
