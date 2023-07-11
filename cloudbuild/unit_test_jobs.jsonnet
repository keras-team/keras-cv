local base = import 'templates/base.libsonnet';
local gpus = import 'templates/gpus.libsonnet';

local image = std.extVar('image');
local tagName = std.extVar('tag_name');
local gcsBucket = std.extVar('gcs_bucket');
local backend = std.extVar('backend');

local unittest = base.BaseTest {
  // Configure job name.
  frameworkPrefix: backend,
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
      export KERAS_BACKEND=jax
      echo $backend
      export JAX_ENABLE_X64=true

      # Run whatever is in `command` here.
      ${@:0}
    |||
  ],
  command: [
    'pytest --run_large --durations 0',
    'keras_cv/bounding_box',
    'keras_cv/callbacks',
    'keras_cv/losses',
    'keras_cv/layers/object_detection',
    'keras_cv/layers/preprocessing',
    'keras_cv/models/backbones',
    'keras_cv/models/classification',
    'keras_cv/models/object_detection/retinanet',
    'keras_cv/models/object_detection/yolo_v8',
  ],
};

std.manifestYamlDoc(unittest.oneshotJob, quote_keys=false)
