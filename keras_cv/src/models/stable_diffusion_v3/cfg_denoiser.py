from keras import ops


class CFGDenoiser:
    def __call__(self, batched, cond_scale):
        # `batched` is the outputs from `BaseModel.apply_model`
        pos_out, neg_out = ops.split(batched, 2, axis=0)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        return scaled
