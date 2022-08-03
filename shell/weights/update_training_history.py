import json
import os
import sys

import tensorboard as tb
from absl import flags

flags.DEFINE_string("tensorboard_logs_path", None, "Path to tensorboard logs to load")
flags.DEFINE_string("training_script_path", None, "Path to the training script")
flags.DEFINE_string(
    "weights_version",
    None,
    "The version of the training script used to produce the latest weights. For example, v0",
)
flags.DEFINE_string(
    "pr_number", None, "The PR number that merged the training script into KerasCV"
)
flags.DEFINE_string(
    "pr_author", None, "The GitHub username of the author of the training script"
)

FLAGS = flags.FLAGS
FLAGS(sys.argv)

weights_version = FLAGS.weights_version or input(
    "Input the weights version for your script\n"
)

training_script_path = FLAGS.training_script_path or input(
    "Input the path to your training script\n"
)
training_script_json_path = training_script_path.replace(".py", ".json")
full_training_script_path = os.path.abspath(training_script_path)

# Build an experiment name structured as dataset/task-version (this experiment name will then match the name of the weights in GCS)
training_script_rooted_at_training = full_training_script_path[
    full_training_script_path.index("keras-cv/examples/training/") + 27 :
]
training_script_dirs = training_script_rooted_at_training.split("/")
tensorboard_experiment_name = (
    f"{training_script_dirs[2]}/{training_script_dirs[0]}-{weights_version}"
)

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

new_results = {
    "validation_accuracy": max_validation_accuracy,
    "epochs_trained": training_epochs,
    "tensorboard_logs": f"https://tensorboard.dev/experiment/{tensorboard_experiment_id}/",
}

pr_number = FLAGS.pr_number or input(
    "Input the PR number that merged your script into KerasCV\n"
)
pr_number = FLAGS.pr_number or input(
    "Input your GitHub username (or the username of the PR author, if you're not the author)\n"
)

# Check if the JSON file already exists
results_file = open(training_script_json_path, "r")
results_string = results_file.read()
results = json.loads(results_string) if results_string != "" else {}
results_file.close()

results_file = open(training_script_json_path, "w")
results[weights_version] = new_results
json.dump(results, results_file)
results_file.close()
