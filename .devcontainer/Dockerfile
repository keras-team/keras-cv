ARG VARIANT

FROM tensorflow/tensorflow:2.9.1${VARIANT}

RUN apt-get update && apt-get install -y \
  locales \
  && rm -rf /var/lib/apt/lists/*

RUN pip install flake8 isort black pytest