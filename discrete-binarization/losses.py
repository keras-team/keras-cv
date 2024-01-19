import keras
from keras import ops


class DiceLoss(keras.losses.Loss):
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, y_true, y_pred, mask, weights=None):
        if weights is not None:
            mask = weights * mask
        intersection = ops.sum((y_pred * y_true * mask))
        union = ops.sum((y_pred * mask)) + ops.sum(y_true * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        return loss


class MaskL1Loss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred, mask):
        mask_sum = ops.sum(mask)
        loss = ops.cond(
            mask_sum == 0,
            lambda: mask_sum,
            lambda: ops.sum(ops.absolute(y_pred - y_true) * mask) / mask_sum,
        )
        return loss


class BalanceCrossEntropyLoss(keras.losses.Loss):
    def __init__(self, negative_ratio=3.0, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, y_true, y_pred, mask, return_origin=False):
        positive = ops.cast(y_true * mask, "uint8")
        negative = ops.cast(((1 - y_true) * mask), "uint8")
        positive_count = int(ops.sum(ops.cast(positive, "float32")))
        negative_count = ops.min(
            int(ops.sum(ops.cast(negative, "float32"))),
            int(positive_count * self.negative_ratio),
        )
        loss = keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction=None,
        )(y_true=y_true, y_pred=y_pred)
        positive_loss = loss * ops.cast(positive, "float32")
        negative_loss = loss * ops.cast(negative, "float32")
        negative_loss, _ = ops.topk(
            ops.reshape(negative_loss, (-1)), negative_count
        )

        balance_loss = (ops.sum(positive_loss) + ops.sum(negative_loss)) / (
            positive_count + negative_count + self.eps
        )

        if return_origin:
            return balance_loss, loss
        return balance_loss


class DBLoss(keras.losses.Loss):
    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5, **kwargs):
        super().__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def call(self, y_true, y_pred, mask):
        p_map_pred, t_map_pred, b_map_pred = ops.split(y_pred, 3, axis=-1)
        shrink_map, thresh_map = y_true
        shrink_mask, thresh_mask = mask

        bce_loss, bce_map = self.bce_loss(
            y_true=shrink_map,
            y_pred=p_map_pred,
            mask=shrink_mask,
            return_origin=True,
        )
        l1_loss = self.l1_loss(
            y_true=thresh_map,
            y_pred=t_map_pred,
            mask=thresh_mask,
        )
        bce_map = (bce_map - ops.minimum(bce_map)) / (
            ops.maximum(bce_map) - ops.maximum(bce_map)
        )
        dice_loss = self.dice_loss(
            y_true=shrink_map,
            y_pred=b_map_pred,
            weights=bce_map + 1,
        )
        loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        return loss
