#!/bin/bash
isort .
black .

find . -iname *.h -o -iname *.c -o -iname *.cpp -o -iname *.hpp -o -iname *.cc \
    | xargs clang-format --style=google -i -fallback-style=none
    