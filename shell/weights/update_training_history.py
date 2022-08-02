import json
import os
import struct
import sys
import tensorboard as tb

from tensorflow.python.summary.summary_iterator import summary_iterator

training_script_path = input("Input the path to your training script\n")
training_script_json_path = training_script_path.replace(".py", ".json")

weights_version = input("Input the weights version for your script\n")

tensorboard_logs_path = input('Input the path to the TensorBoard logs\n')
tensorboard_experiment_id = os.popen(f"tensorboard dev upload --logdir {tensorboard_logs_path} --one_shot --verbose 0").read().split('/')[-2]

tensorboard_experiment = tb.data.experimental.ExperimentFromDev(tensorboard_experiment_id)

tensorboard_results = tensorboard_experiment.get_scalars()

training_epochs = max(tensorboard_results[tensorboard_results.run == "train"].step)
max_validation_accuracy = max(tensorboard_results[(tensorboard_results.run == "validation") & (tensorboard_results.tag == "epoch_accuracy")].value)





###
#
# max_validation_accuracy = None
#
# new_results = {weights_version: {"validation_accuracy": max_validation_accuracy, "tensorboard_logs": f"https://tensorboard.dev/experiment/{tensorboard_experiment_id}}/"}}
#
# # Check if the file already exists
# results_file = open(FLAGS.training_script_json, "r")
# results_string = results_file.read()
# results = json.loads(results_string) if results_string != "" else {}
# results_file.close()
#
# results_file = open(FLAGS.training_script_json, "w")
# results[FLAGS.weights_version] = {"validation_accuracy": max_validation_accuracy}
# json.dump(results, results_file)
# results_file.close()
