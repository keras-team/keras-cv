# KerasCV Accelerators Testing

This `cloudbuild/` directory contains configurations for accelerators (GPU/TPU)
testing. Briefly, for each PR, it copies the PR's code to a base docker image
which contains KerasCV dependencies to make a new docker image, and deploys the
new image to Google Kubernetes Engine cluster, then run all tests in
`keras_cv/` via Google Cloud Build.

- `cloudbuild.yaml`: The cloud build configuration that specifies steps to run
  by cloud build.
- `Dockerfile`: The configuration to build the docker image for deployment.
- `requirements.txt`: Dependencies of KerasCV.
- `unit_test_jobs.jsonnet`: Jsonnet config that tells GKE cluster to run all
  unit tests in `keras_cv/`.

This test is powered by [ml-testing-accelerators](https://github.com/GoogleCloudPlatform/ml-testing-accelerators).


### Adding Test Dependencies

To add a dependency for GPU tests:
- Create a PR adding the dependency to `requirements.txt`
- Have a Keras team member update the Docker image for GPU tests by running the remaining steps
- Create a `Dockerfile` with the following contents:
```
FROM tensorflow/tensorflow:2.9.1-gpu
RUN apt-get -y update
RUN apt-get -y install git
RUN git clone https://github.com/{path_to_keras_cv_fork}.git
RUN cd keras-cv && git checkout {branch_name}
RUN pip install -r keras-cv/cloudbuild/requirements.txt
```
- Run the following command from the directory with your `Dockerfile`:
```
gcloud builds submit --region=us-west1 --tag us-west1-docker.pkg.dev/keras-team-test/keras-cv-test/keras-cv-image:deps --timeout=10m
```
- Merge the PR adding the dependency
