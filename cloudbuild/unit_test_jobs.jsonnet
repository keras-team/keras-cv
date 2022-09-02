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
      # Run whatever is in `command` here.
      ${@:0}
    |||
  ],
  command: [
    'INTEGRATION=true',
    'pytest',
    'keras_cv',
  ],
  env: 
};

std.manifestYamlDoc(unittest.oneshotJob, quote_keys=false)
