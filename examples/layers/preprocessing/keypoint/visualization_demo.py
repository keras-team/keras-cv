# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
   visualization_demo.py is used to visualize the dataset with bounding boxes.
"""
import demo_utils


def main():
    inputs = demo_utils.load_AFLW2000_dataset(keypoint_format="xy")
    demo_utils.visualize_data(inputs)


if __name__ == "__main__":
    main()
