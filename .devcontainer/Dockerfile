ARG VERSION

FROM tensorflow/tensorflow:${VERSION}

RUN apt-get update && apt-get install -y \
  locales \
  && rm -rf /var/lib/apt/lists/*

RUN pip install \
  black[jupyter] \
  flake8 \
  isort \
  pytest \
  regex \
  tensorflow_datasets \
  pycocotools
