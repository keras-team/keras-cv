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

def main():
    batch_size = 9
    dataset, ds_info = load_pascal_voc(
        split="train", bounding_box_format="rel_yxyx", batch_size=batch_size
    )

    for example in dataset.take(1):
        images, boxes = example["images"], example["bounding_boxes"]
        boxes = boxes.to_tensor(default_value=-1)
        print(boxes)
        color = tf.constant(((255.0, 0, 0),))
        plotted_images = tf.image.draw_bounding_boxes(
            images, boxes[..., :4], color, name=None
        )
        plt.figure(figsize=(10, 10))
        for i in range(batch_size):
            plt.subplot(batch_size // 3, batch_size // 3, i + 1)
            plt.imshow(plotted_images[i].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
