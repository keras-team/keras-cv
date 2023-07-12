# KerasCV Accelerators Testing

This `cloudbuild/` directory contains configurations for accelerators (GPU/TPU)
testing. Briefly, for each PR, it copies the PR's code to a base docker image
which contains KerasCV dependencies to make a new docker image, and deploys the
new image to Google Kubernetes Engine cluster, then run all tests in
`keras_cv/` via Google Cloud Build.

- `cloudbuild.yaml`: The cloud build configuration that specifies steps to run
  by cloud build.
- `Dockerfile`: The configuration to build the docker image for deployment.
- `unit_test_jobs.jsonnet`: Jsonnet config that tells GKE cluster to run all
  unit tests in `keras_cv/`.

This test is powered by [ml-testing-accelerators](https://github.com/GoogleCloudPlatform/ml-testing-accelerators).


### Adding Test Dependencies
You must be authorized to run builds in the `keras-team-test` GCP project.
If you are not, please open a GitHub issue and ping a team member.
To authorize yourself with `keras-team-test`, run:

```bash
gcloud config set project keras-team-test
```

To add a dependency for GPU tests:
- Create a PR adding the dependency to `requirements.txt`
- Have a Keras team member update the Docker image for GPU tests by running the remaining steps
- Create a `Dockerfile` with the following contents:
```
FROM tensorflow/tensorflow:2.13.0-gpu
RUN \
    apt-get -y update && \
    apt-get -y install openjdk-8-jdk && \
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add
RUN apt-get -y update
RUN apt-get -y install bazel-5.4.0
RUN apt-get -y install git
RUN git clone https://github.com/{path_to_keras_cv_fork}.git
RUN cd keras-cv && git checkout {branch_name}
RUN pip install -r keras-cv/requirements.txt
```
- Run the following command from the directory with your `Dockerfile`:
```
gcloud builds submit --region=us-west1 --tag us-west1-docker.pkg.dev/keras-team-test/keras-cv-test/keras-cv-image-tensorflow:deps --timeout=20m
```
- Repeat the last two steps for Jax and Torch (replacing "tensorflow" with "jax"
 or "torch" in the docker image target name). `Dockerfile` for jax:
```
FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y git
RUN git clone https://github.com/{path_to_keras_cv_fork}.git
RUN cd keras-cv && git checkout {branch_name}
RUN pip install -r keras-cv/requirements.txt
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
  and for torch:
```
FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y git
RUN git clone https://github.com/{path_to_keras_cv_fork}.git
RUN cd keras-cv && git checkout {branch_name}
RUN pip install -r keras-cv/requirements.txt
RUN pip install torch torchvision
```
- Merge the PR adding the dependency
