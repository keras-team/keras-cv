local base = import 'templates/base.libsonnet';
local gpus = import 'templates/gpus.libsonnet';

local image = std.extVar('image');
local tagName = std.extVar('tag_name');
local gcsBucket = std.extVar('gcs_bucket');

local unittest = base.BaseTest {
  // Configure job name.
  frameworkPrefix: "tf",
  modelName: "keras-cv",
  mode: "unit-tests",
  timeout: 3600, # 1 hour, in seconds

  // Set up runtime environment.
  image: image,
  imageTag: tagName,
  accelerator: gpus.teslaT4,
  outputBucket: gcsBucket,

  entrypoint: [
    'bash',
    '-c',
    |||
      # Build custom ops from source
      python build_deps/configure.py
      bazel build keras_cv/custom_ops:all --verbose_failures
      cp bazel-bin/keras_cv/custom_ops/*.so keras_cv/custom_ops/
      TEST_CUSTOM_OPS=true

      # Run whatever is in `command` here.
      ${@:0}
    |||
  ],
  command: [
    'pytest',
    'keras_cv',
  ],
};

std.manifestYamlDoc(unittest.oneshotJob, quote_keys=false)
