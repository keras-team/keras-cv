import json
import os
import sys

import tensorboard as tb
from absl import flags

flags.DEFINE_string(
    "model_name", None, "The name of the KerasCV.model that was trained"
)
flags.DEFINE_string(
    "tensorboard_logs_path", None, "Path to tensorboard logs to load"
)
flags.DEFINE_string("training_script_path", None, "Path to the training script")
flags.DEFINE_string(
    "script_version",
    None,
    "commit hash of the latest commit in KerasCV/master "
    "for the training script",
)
flags.DEFINE_string(
    "weights_version",
    None,
    "The version of the training script used to produce the latest weights. "
    "For example, v0",
)
flags.DEFINE_string(
    "contributor",
    None,
    "The GitHub username of the contributor of these results",
)
flags.DEFINE_string(
    "accelerators", None, "The number of accelerators used for training."
)

FLAGS = flags.FLAGS
FLAGS(sys.argv)

model_name = FLAGS.model_name or input(
    "Input the name of the KerasCV.model that was trained\n"
)
weights_version = FLAGS.weights_version or input(
    "Input the weights version for your script\n"
)

training_script_path = FLAGS.training_script_path or input(
    "Input the path to your training script\n"
)
full_training_script_path = os.path.abspath(training_script_path)

# Build an experiment name.
# This will be structured as task/training_script_name/model_name-version
training_script_rooted_at_training = full_training_script_path[
    full_training_script_path.index("keras-cv/examples/training/") + 27 :
]
training_script_dirs = training_script_rooted_at_training.split("/")
tensorboard_experiment_name = f"{training_script_dirs[0]}/{'/'.join(training_script_dirs[1:])[:-3]}/{model_name}-{weights_version}"  # noqa: E501

training_script_json_path = full_training_script_path[
    : full_training_script_path.index("keras-cv/examples/training/") + 27
] + "/".join(training_script_dirs[:2] + ["training_history.json"])

script_version = FLAGS.script_version or input(
    "Input the commit hash of the latest commit in KerasCV/master "
    "for the training script used for training."
)

tensorboard_logs_path = FLAGS.tensorboard_logs_path or input(
    "Input the path to the TensorBoard logs\n"
)
tensorboard_experiment_id = (
    os.popen(
        f"python3 -m tensorboard.main dev upload "
        f"--logdir {tensorboard_logs_path} "
        f"--name {tensorboard_experiment_name} "
        f"--one_shot --verbose 0"
    )
    .read()
    .split("/")[-2]
)

tensorboard_experiment = tb.data.experimental.ExperimentFromDev(
    tensorboard_experiment_id
)

tensorboard_results = tensorboard_experiment.get_scalars()

training_epochs = max(
    tensorboard_results[tensorboard_results.run == "train"].step
)

results_tags = tensorboard_results.tag.unique()

# Validation accuracy won't exist in all logs (e.g for OD tasks).
# We capture the max validation accuracy if it exists, but otherwise omit it.
max_validation_accuracy = None
if (
    "epoch_categorical_accuracy" in results_tags
    or "epoch_sparse_categorical_accuracy" in results_tags
):
    max_validation_accuracy = max(
        tensorboard_results[
            (tensorboard_results.run == "validation")
            & (
                (tensorboard_results.tag == "epoch_categorical_accuracy")
                | (
                    tensorboard_results.tag
                    == "epoch_sparse_categorical_accuracy"
                )
            )
        ].value
    )
    max_validation_accuracy = f"{max_validation_accuracy:.4f}"

# Mean IOU won't exist in all logs (e.g for classification tasks).
# We capture the max IOU if it exists, but otherwise omit it.
max_mean_iou = None
if "epoch_mean_io_u" in results_tags:
    max_mean_iou = max(
        tensorboard_results[
            (tensorboard_results.run == "validation")
            & (tensorboard_results.tag == "epoch_mean_io_u")
        ].value
    )
    max_mean_iou = f"{max_mean_iou:.4f}"

contributor = FLAGS.contributor or input(
    "Input your GitHub username "
    "(or the username of the contributor, if it's not you)\n"
)

accelerators = FLAGS.accelerators or input(
    "Input the number of accelerators used during training.\n"
)

args = input(
    "Input any training arguments used for the training script.\n"
    "Use comma-separate, colon-split key-value pairs. For example:\n"
    "arg1:value, arg2:value\n"
)

args_dict = {}
for arg in args.split(","):
    if len(arg.strip()) == 0:
        continue
    key_value_pair = [s.strip() for s in arg.split(":")]
    args_dict[key_value_pair[0]] = key_value_pair[1]

new_results = {
    "script": {
        "name": "/".join(training_script_dirs[2:]),
        "version": script_version,
    },
    "epochs_trained": training_epochs,
    "tensorboard_logs": f"https://tensorboard.dev/experiment/{tensorboard_experiment_id}/",  # noqa: E501
    "contributor": contributor,
    "args": args_dict,
    "accelerators": int(accelerators),
}

if max_validation_accuracy is not None:
    new_results["validation_accuracy"] = max_validation_accuracy

if max_mean_iou is not None:
    new_results["validation_mean_iou"] = max_mean_iou

# Check if the JSON file already exists
results_file = open(training_script_json_path, "r")
results_string = results_file.read()
results = json.loads(results_string) if results_string != "" else {}
results_file.close()

# If we've never run this script on this model, insert a record for it
if model_name not in results:
    results[model_name] = {}

# Add this run's results to the model's record
model_results = results[model_name]
model_results[weights_version] = new_results

# Save the updated results
results_file = open(training_script_json_path, "w")
json.dump(results, results_file, indent=4, sort_keys=True)
results_file.close()
