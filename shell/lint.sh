#!/bin/bash
files="."
if [ ! -z "$1" ]
  then
    files=$(eval $1)
    if [ -z "$files" ]
      then
        exit 0
    fi
fi

isort -c $files
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
  exit 1
fi
[ -z "$1" ] && echo "no issues with isort"
flake8 $files
if ! [ $? -eq 0 ]
then
  echo "Please fix the code style issue."
  exit 1
fi
[ -z "$1" ] && echo "no issues with flake8"
black --check $files
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
    exit 1
fi
[ -z "$1" ] && echo "no issues with black"
for i in $(find examples keras_cv -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo "Copyright not found in $i"
    exit 1
  fi
done
echo "linting success!"
