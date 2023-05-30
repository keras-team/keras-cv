## How to contribute code

Follow these steps to submit your code contribution.

[You can find a list of issues that we are looking for contributors on here!](https://github.com/keras-team/keras-cv/labels/contribution-welcome)

### Step 1. Open an issue

Before making any changes, we recommend opening an issue (if one doesn't already
exist) and discussing your proposed changes. This way, we can give you feedback
and validate the proposed changes.

If your code change involves the fixing of a bug, please include a
[Colab](https://colab.research.google.com/) notebook that shows
how to reproduce the broken behavior.

If the changes are minor (simple bug fix or documentation fix), then feel free
to open a PR without discussion.

### Step 2. Make code changes

To make code changes, you need to fork the repository. You will need to set up a
development environment and run the unit tests. This is covered in section
"set up environment".

If your code change involves introducing a new API change, please see our
[API Design Guidelines](API_DESIGN.md).

**Notes**

- Make sure to add a new entry to [serialization tests](https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/serialization_test.py#L37) for new layers.

### Step 3. Create a pull request

Once the change is ready, open a pull request from your branch in your fork to
the master branch in [keras-team/keras-cv](https://github.com/keras-team/keras-cv).

### Step 4. Sign the Contributor License Agreement

After creating the pull request, you will need to sign the Google CLA agreement.
The agreement can be found at [https://cla.developers.google.com/clas](https://cla.developers.google.com/clas).


### Step 5. Code review

CI tests will automatically be run directly on your pull request. Their
status will be reported back via GitHub actions.

There may be
several rounds of comments and code changes before the pull request gets
approved by the reviewer.

![Approval from reviewer](https://i.imgur.com/zgRziTt.png)

### Step 6. Merging

Once the pull request is approved, a team member will take care of merging.

## Contributing models
When contributing new models, please validate model performance by providing training results. You can do this using our existing [ImageNet training script](https://github.com/keras-team/keras-cv/blob/master/examples/training/classification/imagenet/basic_training.py) or by contributing a custom training script of your own (see "Contributing training scripts" below). Training results can be added to the training history log with [this script](https://github.com/keras-team/keras-cv/blob/master/shell/weights/update_training_history.py), or shared with the team via Google Drive (we'll need TensorBoard logs as well as weights). Either way, the KerasCV team will need to upload the weights to our GCS bucket for distribution.

For an initial submission, trained weights do not need to exactly match paper-claimed results. As a baseline, let's shoot for 90% of the paper-claimed ImageNet top-1 accuracy. However, we should strive to improve these weights quickly to at least match paper-claimed results.

## Contributing training scripts

KerasCV is working to include a catalog of high-performing model training scripts for the models included in KerasCV.models and is welcoming contributions for these scripts. These training scripts serve as documentation of good training techniques and will be used to train weights that will be offered in KerasCV models through the package.

The KerasCV team will run submitted training scripts to produce weights for KerasCV, and will attribute strong weights to contributors via a training script ranking system. Stay tuned for more details about that.

Incremental improvements to existing training scripts are welcome, provided that they come with evidence of improved validation performance.

You can also open an issue to add weights for a specific model using a pre-existing script! In your issue, provide your training logs and resulting weights. Specify the arguments that were used to run the script, and provide support for those choices. If your weights beat our current weights, they'll become our default pre-trained weights for your model/task in KerasCV.models!

To contribute a new script, start by opening an issue and tagging @ianstenbit to discuss the task, dataset, and/or model for which you'd like to add a script. Once they've taken a look, you can prepare a PR to introduce the new training script.

See [this example script](https://github.com/keras-team/keras-cv/blob/master/examples/training/classification/imagenet/basic_training.py) for training ImageNet classification. Please follow the structure of this training script in contributing your own script. New scripts should either:
- Train a task for which we don't have a training script already
- Include a meaningfully different training approach for a given task
- Introduce a custom training method for a specific model or dataset, based on empirical evidence of efficacy.

When contributing training scripts or proposing runs, please include documentation to support decisions about training including hyperparameter choices. Examples of good documentation would be recent literature or a reference to a hyperparameter search.

Our default training scripts train using ImageNet. Because we cannot distribute this dataset, you will need to modify your dataloading step to load the dataset on your system if you wish to run training yourself. You are also welcome to locally train against a different dataset, provided that you include documentation in your PR supporting the claim that your script will still perform well against ImageNet.

We look forward to delivering great pre-trained models in KerasCV with the help of your contributions!

## Contributing custom ops

We do not plan to accept contributed custom ops due to the maintenance burden that they introduce. If there is a clear need for a specific custom op that should live in KerasCV, please consult the KerasCV team before implementing it, as we expect to reject contributions of custom ops by default.

We currently support only a small handful of ops that run on CPU and are not used at inference time.

If you are updating existing custom ops, you can re-compile the binaries from source using the instructions in the `Tests that require custom ops` section below.

## set up environment

Setting up your KerasCV development environment requires you to fork the KerasCV repository,
clone the repository, install dependencies, and execute `python setup.py develop`.

You can achieve this by running the following commands:

```shell
gh repo fork keras-team/keras-cv --clone --remote
cd keras-cv
pip install ".[tests]"
python setup.py develop
```

The first line relies on having an installation of [the GitHub CLI](https://github.com/cli/cli).

Following these commands you should be able to run the tests using `pytest keras_cv`.
Please report any issues running tests following these steps.

Note that this will _not_ install custom ops. If you'd like to install custom ops from source, you can compile the binaries and add them to your local environment manually (requires Bazel):

```shell
python build_deps/configure.py

bazel build keras_cv/custom_ops:all
mv bazel-bin/keras_cv/custom_ops/*.so keras_cv/custom_ops
```

## Run tests

KerasCV is tested using [PyTest](https://docs.pytest.org/en/6.2.x/).

### Run a test file

To run a test file, run `pytest path/to/file` from the root directory of keras\_cv.

### Run a single test case

To run a single test, you can use `-k=<your_regex>`
to use regular expression to match the test you want to run. For example, you
can use the following command to run all the tests in `cut_mix_test.py`,
whose names contain `label`,

```
pytest keras_cv/layers/preprocessing/cut_mix_test.py -k="label"
```

### Run all tests

You can run the unit tests for KerasCV by running:
```
pytest keras_cv/
```

### Tests that require custom ops
For tests that require custom ops, you'll have to compile the custom ops and make them available to your local Python code:
```shell
python build_deps/configure.py
bazel build keras_cv/custom_ops:all
cp bazel-bin/keras_cv/custom_ops/*.so keras_cv/custom_ops/
```

Tests which use custom ops are disabled by default, but can be run by setting the environment variable `TEST_CUSTOM_OPS=true`.

## Formatting the Code
We use `flake8`, `isort`, `black` and `clang-format` for code formatting. You can run
the following commands manually every time you want to format your code:

- Run `shell/format.sh` to format your code
- Run `shell/lint.sh` to check the result.

If after running these the CI flow is still failing, try updating `flake8`, `isort`, `black` and `clang-format`.
This can be done by running `pip install --upgrade black`, `pip install --upgrade flake8`,
`pip install --upgrade isort` and `pip install --upgrade clang-format`

Note: The linting checks could be automated activating  
      pre-commit hooks with `git config core.hooksPath .github/.githooks`

## Community Guidelines

This project follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/).
