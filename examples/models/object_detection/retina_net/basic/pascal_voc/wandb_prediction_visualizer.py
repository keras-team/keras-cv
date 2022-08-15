import wandb

import tensorflow as tf
from keras_cv import bounding_box

class WandbTablesBuilder:
    """
    Utility class that contains useful methods to create W&B Tables,
    and log it to W&B.
    """
    def init_data_table(self, column_names: list):
        """Initialize the W&B Tables for validation data.
        Call this method `on_train_begin` or equivalent hook. This is followed by
        adding data to the table row or column wise.

        Args:
            column_names (list): Column names for W&B Tables.
        """
        self.data_table = wandb.Table(columns=column_names, allow_mixed_types=True)

    def init_pred_table(self, column_names: list):
        """Initialize the W&B Tables for model evaluation.
        Call this method `on_epoch_end` or equivalent hook. This is followed by
        adding data to the table row or column wise.

        Args:
            column_names (list): Column names for W&B Tables.
        """
        self.pred_table = wandb.Table(columns=column_names)

    def log_data_table(self,
                    name: str='val',
                    type: str='dataset',
                    table_name: str='val_data'):
        """Log the `data_table` as W&B artifact and call
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded data (images, text, scalar, etc.).
        This allows the data to be uploaded just once.

        Args:
            name (str):  A human-readable name for this artifact, which is how
                you can identify this artifact in the UI or reference
                it in use_artifact calls. (default is 'val')
            type (str): The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'val_data')
            table_name (str): The name of the table as will be displayed in the UI.
        """
        data_artifact = wandb.Artifact(name, type=type)
        data_artifact.add(self.data_table, table_name)

        # Calling `use_artifact` uploads the data to W&B.
        wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        # We get the reference table.
        self.data_table_ref = data_artifact.get(table_name)

    def log_pred_table(self,
                    type: str='evaluation',
                    table_name: str='eval_data'):
        """Log the W&B Tables for model evaluation.
        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.

        Args:
            type (str): The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'val_data')
            table_name (str): The name of the table as will be displayed in the UI.
        """
        pred_artifact = wandb.Artifact(
            f'run_{wandb.run.id}_pred', type=type)
        pred_artifact.add(self.pred_table, table_name)
        # TODO: Add aliases
        wandb.run.log_artifact(pred_artifact)


class WandbPredictionVisualizer(tf.keras.callbacks.Callback):
    """
    The RetinaNetVizCallback:
      - logs the validation data or ground truth as W&B Tables,
      - performs inference to get model prediction `on_epoch_end`,
      - and logs the predictions as W&B Artifacts `on_epoch_end`,
      - it uses referencing thus data is uploaded just once.

    Example:
        ```
        run = wandb.init(project='...')
        ...
        model.fit(
            ...,
            callbacks=[
                RetinaNetVizWandbCallback(
                    val_ds,
                    val_dataset_info,
                    num_classes=20,
                    bounding_box_format="xywh",
                    num_samples=10
                )
            ]
        )
        ```

    Args:
        data: A dataloader built using `tf.data` API. The callback
            expects the dataloader to return a tuple of images and bounding boxes.
        dataset_info: It is the `tfds.core.DatasetInfo` containing the info associated
            with the dataset.
        num_claslses (int): Number of classes. Defaults to 80.
        bounding_box_format (str): String representing the bounding box format used by
            the object detector. Check out `keras_cv.bounding_box.convert_format()` to
            get more info on different allowed formats.
        num_samples (int): The number of samples you want to visualize.
        confidence_threshold (float): Use this to filter the predicted bounding boxes
            below the threshold before logging to W&B.

    """

    def __init__(
        self,
        data,
        dataset_info,
        num_classes,
        bounding_box_format,
        num_samples=100,
    ):
        super().__init__()
        self.bounding_box_format = bounding_box_format
        self.num_samples = num_samples

        # Make unbatched iterator from `tf.data.Dataset`.
        self.val_ds = data.unbatch().take(self.num_samples)

        # A dictionary mapping class id to class label.
        self.int2str = dataset_info.features["objects"]["label"].int2str
        self.class_id_to_label = {idx: self.int2str(idx) for idx in range(num_classes)}

        # When logging bounding boxes or segmentation masks along with W&B Tables,
        # a `wandb.Classes` instance is passed to `wandb.Image`.
        self.class_set = wandb.Classes(
            [
                {"id": idx, "name": label}
                for idx, label in self.class_id_to_label.items()
            ]
        )

        self.tables_builder = WandbTablesBuilder()

    def on_train_begin(self, logs=None):
        # Initialize W&B table to log validation data
        self.tables_builder.init_data_table(
            column_names = ["image_index", "ground_truth"]
        )
        # Add validation data to the table
        self.add_ground_truth()
        # Log the table to W&B
        self.tables_builder.log_data_table()

    def on_epoch_end(self, epoch, logs=None):
        # Initialize a prediction wandb table
        self.tables_builder.init_pred_table(
            column_names = ["epoch", "image_index",
                            "ground_truth", "prediction"]
        )
        # Add prediction to the table
        self.add_model_predictions(epoch)
        # Log the eval table to W&B
        self.tables_builder.log_pred_table()

    def add_ground_truth(self):
        """Logic for adding validation/training data to `data_table`.
        This method is called once `on_train_begin` or equivalent hook.
        """
        # Iterate through the samples and log them to the data_table.
        for idx, (image, bboxes) in enumerate(self.val_ds.as_numpy_iterator()):
            # The last element in the bboxes is the label_id.
            assert bboxes.shape[-1] == 5
            bboxes = self._convert_bbox_format(bboxes, image).numpy()

            # Get bounding box formatted for logging to W&B.
            wandb_bboxes = {
                "ground_truth": self._get_wandb_bboxes(bboxes[:, :-1], bboxes[:, -1])
            }

            # Log a row to the data table.
            self.tables_builder.data_table.add_data(
                idx,
                wandb.Image(image, boxes=wandb_bboxes, classes=self.class_set),
            )

    def add_model_predictions(self, epoch):
        # Get predicted detections
        pred_bboxes, pred_labels, pred_confs = self._infer()

        # Iterate through the samples.
        data_table_ref = self.tables_builder.data_table_ref
        table_idxs = data_table_ref.get_index()
        assert len(table_idxs) == len(pred_bboxes)

        for idx in table_idxs:
            pred_bbox, pred_label, pred_conf = (
                pred_bboxes[idx],
                pred_labels[idx],
                pred_confs[idx],
            )

            # Get dict of bounding boxes in the format required by `wandb.Image`.
            wandb_bboxes = {
                "predictions": self._get_wandb_bboxes(
                    pred_bbox, pred_label, log_gt=False, conf_scores=pred_conf
                )
            }

            # Log a row to the eval table.
            self.tables_builder.pred_table.add_data(
                epoch,
                data_table_ref.data[idx][0],
                data_table_ref.data[idx][1],
                wandb.Image(
                    data_table_ref.data[idx][1],
                    boxes=wandb_bboxes,
                    classes=self.class_set,
                ),
            )

    def _infer(self):
        pred_bboxes, pred_labels, pred_confs = [], [], []

        for idx, (image, bboxes) in enumerate(self.val_ds.as_numpy_iterator()):
            assert image.ndim == 3

            # Get model prediction.
            pred = self.model(tf.expand_dims(image, axis=0))
            pred = tf.squeeze(pred["inference"], axis=0).numpy()

            if pred.ndim == 1:
                bbox_preds.append([])
                pred_labels.append([])
                pred_confs.append([])
            else:
                pred_bbox, pred_label, pred_conf = (
                    pred[:, :4],
                    pred[:, 4],
                    pred[:, -1],
                )

                pred_bbox = self._convert_bbox_format(pred_bbox, image).numpy()
                pred_bboxes.append(pred_bbox)
                pred_labels.append(pred_label)
                pred_confs.append(pred_conf)

        return pred_bboxes, pred_labels, pred_confs

    def _get_wandb_bboxes(self, bboxes, label_ids, log_gt=True, conf_scores=None):
        """
        Return a dict of bounding boxes in the format required by `wandb.Image`
        to log bounding boxes to W&B.

        To learn about the format check out the docs:
        https://docs.wandb.ai/guides/track/log/media#image-overlays
        """
        assert len(bboxes) == len(label_ids)

        box_data = []
        # TODO (ayulockin): filter results by a threshold.
        for i, (bbox, label_id) in enumerate(zip(bboxes, label_ids)):
            # "rel_xyxy" format
            position = dict(
                minX=float(bbox[0]),
                minY=float(bbox[1]),
                maxX=float(bbox[2]),
                maxY=float(bbox[3]),
            )

            box_dict = {
                "position": position,
                "class_id": int(label_id),
                "box_caption": self.class_id_to_label[label_id],
            }

            if not log_gt:
                if conf_scores is not None:
                    score = conf_scores[i]
                    caption = f"{self.class_id_to_label[label_id]}|{float(score)}"
                    box_dict["box_caption"] = caption

            box_data.append(box_dict)

        wandb_bboxes = {"box_data": box_data, "class_labels": self.class_id_to_label}

        return wandb_bboxes

    def _convert_bbox_format(self, bbox, image):
        # Convert the bounding box format to "rel_xyxy".
        # In this format, the axes are the same as `"xyxy"` but the x
        # coordinates are normalized using the image width, and the y axes the image
        # height. All values in `rel_xyxy` are in the range (0, 1).
        bbox = bounding_box.convert_format(
            bbox, self.bounding_box_format, "rel_xyxy", images=image
        )

        return bbox
