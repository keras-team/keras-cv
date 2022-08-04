import json
import os
import sys

import tensorboard as tb
from absl import flags

flags.DEFINE_string(
    "model_name", None, "The name of the KerasCV.model that was trained"
)
flags.DEFINE_string("tensorboard_logs_path", None, "Path to tensorboard logs to load")
flags.DEFINE_string("training_script_path", None, "Path to the training script")
flags.DEFINE_string(
    "weights_version",
    None,
    "The version of the training script used to produce the latest weights. For example, v0",
)
flags.DEFINE_string(
    "contributor", None, "The GitHub username of the contributor of these results"
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
training_script_json_path = training_script_path.replace(".py", ".json")
full_training_script_path = os.path.abspath(training_script_path)

# Build an experiment name structured as task/training_script_name/model_name-version
training_script_rooted_at_training = full_training_script_path[
    full_training_script_path.index("keras-cv/examples/training/") + 27 :
]
training_script_dirs = training_script_rooted_at_training.split("/")
tensorboard_experiment_name = f"{training_script_dirs[0]}/{training_script_dirs[1][:-3]}/{model_name}-{weights_version}"

tensorboard_logs_path = FLAGS.tensorboard_logs_path or input(
    "Input the path to the TensorBoard logs\n"
)
tensorboard_experiment_id = (
    os.popen(
        f"tensorboard dev upload --logdir {tensorboard_logs_path} --name {tensorboard_experiment_name} --one_shot --verbose 0"
    )
    .read()
    .split("/")[-2]
)

tensorboard_experiment = tb.data.experimental.ExperimentFromDev(
    tensorboard_experiment_id
)

tensorboard_results = tensorboard_experiment.get_scalars()

training_epochs = max(tensorboard_results[tensorboard_results.run == "train"].step)
max_validation_accuracy = max(
    tensorboard_results[
        (tensorboard_results.run == "validation")
        & (tensorboard_results.tag == "epoch_accuracy")
    ].value
)
max_validation_accuracy = f"{max_validation_accuracy:.4f}"

contributor = FLAGS.contributor or input(
    "Input your GitHub username (or the username of the contributor, if it's not you)\n"
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
    "validation_accuracy": max_validation_accuracy,
    "epochs_trained": training_epochs,
    "tensorboard_logs": f"https://tensorboard.dev/experiment/{tensorboard_experiment_id}/",
    "contributor": contributor,
    "args": args_dict,
}

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
