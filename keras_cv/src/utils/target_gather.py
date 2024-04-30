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

from keras_cv.src.backend import ops


def _target_gather(
    targets,
    indices,
    mask=None,
    mask_val=0.0,
):
    """A utility function wrapping tf.gather, which deals with:
     1) both batched and unbatched `targets`
     2) when unbatched `targets` have empty rows, the result will be filled
        with `mask_val`
     3) target masking.

    Args:
     targets: [N, ...] or [batch_size, N, ...] Tensor representing targets such
        as boxes, keypoints, etc.
     indices: [M] or [batch_size, M] int32 Tensor representing indices within
        `targets` to gather.
     mask: optional [M, ...] or [batch_size, M, ...] boolean Tensor representing
        the masking for each target. `True` means the corresponding entity
        should be masked to `mask_val`, `False` means the corresponding
        entity should be the target value.
     mask_val: optional float representing the masking value if `mask` is True
        on the entity.

    Returns:
     targets: [M, ...] or [batch_size, M, ...] Tensor representing
       selected targets.

     Raise:
       ValueError: If `targets` is higher than rank 3.
    """
    targets_shape = list(targets.shape)
    if len(targets_shape) > 3:
        raise ValueError(
            "`target_gather` does not support `targets` with rank "
            "larger than 3, got {}".format(len(targets.shape))
        )

    def _gather_unbatched(labels, match_indices, mask, mask_val):
        """Gather based on unbatched labels and boxes."""
        num_gt_boxes = labels.shape[0]

        def _assign_when_rows_empty():
            if len(labels.shape) > 1:
                mask_shape = [match_indices.shape[0], labels.shape[-1]]
            else:
                mask_shape = [match_indices.shape[0]]
            return ops.cast(mask_val, labels.dtype) * ops.ones(
                mask_shape, dtype=labels.dtype
            )

        def _assign_when_rows_not_empty():
            targets = ops.take(labels, match_indices, axis=0)
            if mask is None:
                return targets
            else:
                masked_targets = ops.cast(
                    mask_val, labels.dtype
                ) * ops.ones_like(mask, dtype=labels.dtype)
                return ops.where(mask, masked_targets, targets)

        if num_gt_boxes > 0:
            return _assign_when_rows_not_empty()
        else:
            return _assign_when_rows_empty()

    def _gather_batched(labels, match_indices, mask, mask_val):
        """Gather based on batched labels."""
        batch_size = labels.shape[0]
        if batch_size == 1:
            if mask is not None:
                result = _gather_unbatched(
                    ops.squeeze(labels, axis=0),
                    ops.squeeze(match_indices, axis=0),
                    ops.squeeze(mask, axis=0),
                    mask_val,
                )
            else:
                result = _gather_unbatched(
                    ops.squeeze(labels, axis=0),
                    ops.squeeze(match_indices, axis=0),
                    None,
                    mask_val,
                )
            return ops.expand_dims(result, axis=0)
        else:
            targets = ops.take_along_axis(
                labels, ops.expand_dims(match_indices, axis=-1), axis=1
            )

            if mask is None:
                return targets
            else:
                masked_targets = ops.cast(
                    mask_val, labels.dtype
                ) * ops.ones_like(mask, dtype=labels.dtype)
                return ops.where(mask, masked_targets, targets)

    if len(targets_shape) <= 2:
        return _gather_unbatched(targets, indices, mask, mask_val)
    elif len(targets_shape) == 3:
        return _gather_batched(targets, indices, mask, mask_val)
