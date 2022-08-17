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
