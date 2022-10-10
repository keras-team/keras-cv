ARG VARIANT

FROM tensorflow/tensorflow:2.10.0${VARIANT}

RUN apt-get update && apt-get install -y \
  locales \
  && rm -rf /var/lib/apt/lists/*

RUN pip install \
  black \
  flake8 \
  isort \
  pytest \
  regex \
  tensorflow_datasets
