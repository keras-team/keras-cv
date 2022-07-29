import struct
import sys
import os
import json

from tensorflow.python.summary.summary_iterator import summary_iterator
from absl import flags

flags.DEFINE_string("logs_path", None, "Path of TensorBoard validation logs to load")
flags.DEFINE_string("training_script_json", None, "Path to the JSON file for run history for the producing training script")
flags.DEFINE_string("weights_version", None, "The version of these weights (e.g. 'v0')")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

# Find the single validation logs in the supplied logs dir
files = [filename for _, _, filename in os.walk(FLAGS.logs_path)]
assert len(files) == 1
assert len(files[0]) == 1
filename = FLAGS.logs_path+'/'+files[0][0]

events = [e for e in summary_iterator(filename)]

max_validation_accuracy = 0
for event in events[1:]:
    assert len(event.summary.value) == 1
    if not event.summary.value[0].tag == "evaluation_accuracy_vs_iterations":
        continue
    validation_accuracy = struct.unpack('f',event.summary.value[0].tensor.tensor_content)[0]
    max_validation_accuracy = max(max_validation_accuracy, validation_accuracy)

max_validation_accuracy = f"{max_validation_accuracy:.4f}"

print(f"Max validation accuracy: {max_validation_accuracy}")

new_results = {FLAGS.weights_version: {"validation_accuracy": max_validation_accuracy}}

# Check if the file already exists
results_file = open(FLAGS.training_script_json, 'r')
results_string = results_file.read()
results = json.loads(results_string) if results_string != "" else {}
results_file.close()

results_file = open(FLAGS.training_script_json, 'w')
results[FLAGS.weights_version] = {"validation_accuracy": max_validation_accuracy}
json.dump(results, results_file)
results_file.close()
