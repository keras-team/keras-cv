# Copyright 2023 The KerasCV Authors
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

import itertools
import os
import pathlib

import numpy as np
import pytest
from absl.testing import parameterized

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.backbones.vit_det.vit_det_aliases import ViTDetBBackbone
from keras_cv.models.segmentation.segment_anything.sam import (
    SegmentAnythingModel,
)
from keras_cv.models.segmentation.segment_anything.sam_layers import (
    TwoWayMultiHeadAttention,
)
from keras_cv.models.segmentation.segment_anything.sam_mask_decoder import (
    SAMMaskDecoder,
)
from keras_cv.models.segmentation.segment_anything.sam_prompt_encoder import (
    SAMPromptEncoder,
)
from keras_cv.models.segmentation.segment_anything.sam_transformer import (
    TwoWayTransformer,
)
from keras_cv.tests.test_case import TestCase


class SAMTest(TestCase):
    def setUp(self):
        self.image_encoder = ViTDetBBackbone()
        self.prompt_encoder = SAMPromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        )
        self.mask_decoder = SAMMaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2, embed_dim=256, mlp_dim=2048, num_heads=8
            ),
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

    def get_prompts(self, B, prompts="all"):
        rng = np.random.default_rng(0)

        prompts_dict = {}

        if "all" in prompts or "points" in prompts:
            prompts_dict["points"] = ops.convert_to_tensor(
                rng.integers(0, 1023, (B, 10, 2)), dtype="float32"
            )
            prompts_dict["labels"] = ops.convert_to_tensor(
                1 * (rng.random((B, 10)) > 0.5), dtype="int32"
            )

        if "all" in prompts or "boxes" in prompts:
            x1y1 = rng.integers(0, 1022, (B, 2))
            x2y2 = rng.integers(x1y1, 1023, (B, 2))
            box = np.stack([x1y1, x2y2], axis=1)
            prompts_dict["boxes"] = ops.convert_to_tensor(
                box[:, None, ...], dtype="float32"
            )
        if "all" in prompts or "masks" in prompts:
            prompts_dict["masks"] = ops.convert_to_tensor(
                1.0 * (rng.random((B, 1, 256, 256, 1)) > 0.5), dtype="float32"
            )

        return prompts_dict

    def test_prompt_encoder_simple(self):
        outputs = self.prompt_encoder(self.get_prompts(7))
        sparse_embeddings, dense_embeddings, dense_positional_embeddings = (
            outputs["sparse_embeddings"],
            outputs["dense_embeddings"],
            outputs["dense_positional_embeddings"],
        )

        trainable_parameters = np.sum(
            [np.prod(x.shape) for x in self.prompt_encoder.trainable_weights]
        )
        num_parameters = np.sum(
            [np.prod(x.shape) for x in self.prompt_encoder.weights]
        )

        sparse_embeddings = ops.convert_to_numpy(sparse_embeddings)
        dense_embeddings = ops.convert_to_numpy(dense_embeddings)
        dense_positional_embeddings = ops.convert_to_numpy(
            dense_positional_embeddings
        )

        self.assertEqual(sparse_embeddings.shape, (7, 12, 256))
        self.assertEqual(dense_embeddings.shape, (7, 64, 64, 256))
        self.assertEqual(dense_positional_embeddings.shape, (1, 64, 64, 256))
        self.assertEqual(trainable_parameters, 6_220)
        self.assertEqual(num_parameters, 6_476)

    @parameterized.named_parameters(
        [
            ("_".join(x), x)
            for x in itertools.chain(
                itertools.combinations(["points", "boxes", "masks"], 1),
                itertools.combinations(["points", "boxes", "masks"], 2),
            )
        ]
    )
    def test_prompt_encoder_partial_prompts(self, prompts):
        prompts_dict = self.get_prompts(7, prompts)
        outputs = self.prompt_encoder(prompts_dict)
        sparse_embeddings, dense_embeddings = (
            outputs["sparse_embeddings"],
            outputs["dense_embeddings"],
        )

        sparse_embeddings_dim = 0
        if "points" in prompts:
            sparse_embeddings_dim += prompts_dict["points"].shape[1]
        if "boxes" in prompts:
            sparse_embeddings_dim += prompts_dict["boxes"].shape[1] * 2
        self.assertAllEqual(
            sparse_embeddings.shape,
            (7, sparse_embeddings_dim, 256),
        )
        self.assertAllEqual(dense_embeddings.shape, (7, 64, 64, 256))
        if "masks" not in prompts:
            no_mask_embed = ops.broadcast_to(
                self.prompt_encoder.no_mask_embed(ops.arange(1)),
                (7, 64, 64, 256),
            )
            self.assertAllClose(dense_embeddings, no_mask_embed)

    def test_two_way_multi_head_attention(self):
        image_embeddings = np.random.randn(1, 64, 64, 256).astype(np.float32)

        prompt_encoder_outputs = self.prompt_encoder(self.get_prompts(1))
        sparse_embeddings = prompt_encoder_outputs["sparse_embeddings"]

        two_way_attention = TwoWayMultiHeadAttention(
            num_heads=8,
            key_dim=256 // 8,
            mlp_dim=2048,
            skip_first_layer_pe=False,
        )
        queries, keys = two_way_attention(
            queries=sparse_embeddings,
            keys=ops.reshape(image_embeddings, (1, 64 * 64, 256)),
            query_pe=sparse_embeddings,
            key_pe=ops.reshape(
                prompt_encoder_outputs["dense_positional_embeddings"],
                (1, 64 * 64, 256),
            ),
        )

        queries, keys = map(ops.convert_to_numpy, [queries, keys])

        self.assertEqual(queries.shape, (1, 12, 256))
        self.assertEqual(keys.shape, (1, 64 * 64, 256))

    def test_two_way_transformer(self):
        prompt_encoder_outputs = self.prompt_encoder(self.get_prompts(1))
        sparse_embeddings = prompt_encoder_outputs["sparse_embeddings"]
        image_embeddings = np.random.randn(1, 64, 64, 256)
        two_way_transformer = TwoWayTransformer(
            depth=2, embed_dim=256, num_heads=8, mlp_dim=2048
        )
        queries, keys = two_way_transformer(
            image_embedding=image_embeddings,
            image_pe=prompt_encoder_outputs["dense_positional_embeddings"],
            point_embedding=sparse_embeddings,
        )

        queries, keys = map(ops.convert_to_numpy, [queries, keys])

        self.assertEqual(queries.shape, (1, 12, 256))
        self.assertEqual(keys.shape, (1, 64 * 64, 256))

    def test_mask_decoder(self):
        prompt_encoder_outputs = self.prompt_encoder(self.get_prompts(1))
        sparse_embeddings, dense_embeddings, dense_positional_embeddings = (
            prompt_encoder_outputs["sparse_embeddings"],
            prompt_encoder_outputs["dense_embeddings"],
            prompt_encoder_outputs["dense_positional_embeddings"],
        )
        image_embeddings = np.random.randn(1, 64, 64, 256)
        outputs = self.mask_decoder(
            dict(
                image_embeddings=image_embeddings,
                image_pe=dense_positional_embeddings,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
        )
        masks, iou_pred = outputs["masks"], outputs["iou_pred"]
        num_parameters = np.sum(
            [np.prod(x.shape) for x in self.mask_decoder.weights]
        )
        masks, iou_pred = map(ops.convert_to_numpy, [masks, iou_pred])
        self.assertEqual(masks.shape, (1, 4, 256, 256))
        self.assertEqual(iou_pred.shape, (1, 4))
        self.assertEqual(num_parameters, 4_058_340)

    @pytest.mark.large
    def test_end_to_end_model_predict(self):
        model = SegmentAnythingModel(
            backbone=self.image_encoder,
            prompt_encoder=self.prompt_encoder,
            mask_decoder=self.mask_decoder,
        )

        mask_prompts = self.get_prompts(1)
        inputs = {
            "images": np.ones((1, 1024, 1024, 3)),
        }
        inputs.update(mask_prompts)

        # Check the number of parameters
        num_parameters = np.sum([np.prod(x.shape) for x in model.weights])
        self.assertEqual(num_parameters, 89_670_912 + 6_476 + 4_058_340)

        # Forward pass through the model
        outputs = model.predict(inputs)
        masks, iou_pred = outputs["masks"], outputs["iou_pred"]

        # Check the output is equal to the one we expect if we
        # run each component separately. This is to confirm that
        # the graph is getting compiled correctly i.e. the jitted
        # execution is equivalent to the eager execution.
        features = self.image_encoder(inputs["images"])
        outputs_ex = self.prompt_encoder(
            {k: v for k, v in inputs.items() if k != "images"}
        )
        outputs_ex = self.mask_decoder(
            {
                "image_embeddings": features,
                "image_pe": outputs_ex["dense_positional_embeddings"],
                "sparse_prompt_embeddings": outputs_ex["sparse_embeddings"],
                "dense_prompt_embeddings": outputs_ex["dense_embeddings"],
            },
        )
        masks_ex, iou_pred_ex = outputs_ex["masks"], outputs_ex["iou_pred"]

        self.assertAllClose(masks, masks_ex, atol=5e-5)
        self.assertAllClose(iou_pred, iou_pred_ex, atol=5e-5)

    @pytest.mark.extra_large
    def test_end_to_end_model_save(self):
        # Build the model
        model = SegmentAnythingModel(
            backbone=self.image_encoder,
            prompt_encoder=self.prompt_encoder,
            mask_decoder=self.mask_decoder,
        )

        mask_prompts = self.get_prompts(1)
        inputs = {
            "images": np.ones((1, 1024, 1024, 3)),
        }
        inputs.update(mask_prompts)

        # Forward pass
        outputs = model(inputs)

        # Save the model
        save_path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(save_path, save_format="keras_v3")
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, SegmentAnythingModel)

        # Check that output matches.
        restored_outputs = restored_model(inputs)
        self.assertAllClose(outputs, restored_outputs)

    @pytest.mark.large
    def test_end_to_end_model_preset(self):
        # Define the RNG. Don't change the seed. This seed
        # was used to generate the inputs for the reference
        # values.
        rng = np.random.default_rng(0)

        # Generate the inputs
        inputs = {
            "images": 255.0 * rng.random((1, 1024, 1024, 3), dtype=np.float32),
            "points": np.array(
                [[[10, 10], [100, 100], [500, 500]]], dtype=np.float32
            ),
            "labels": np.array([[0, 1, 0]], dtype=np.float32),
            "boxes": np.array(
                [[[[10.0, 10.0], [100.0, 100.0]]]], dtype=np.float32
            ),
            "masks": (rng.random((1, 1, 256, 256, 1)) > 0.5).astype(np.float32),
        }

        # Run the model
        model = SegmentAnythingModel.from_preset("sam_base_sa1b")
        outs = model.predict(inputs)

        # Make sure the weights have been loaded correctly.
        masks_expected = np.load(
            pathlib.Path(__file__).parent / "data" / "sam_base_out_masks.npy"
        )
        iou_pred_expected = np.load(
            pathlib.Path(__file__).parent / "data" / "sam_base_out_iou_pred.npy"
        )
        self.assertAllClose(outs["masks"], masks_expected, atol=1e-2, rtol=1e-2)
        self.assertAllClose(
            outs["iou_pred"], iou_pred_expected, atol=1e-2, rtol=1e-2
        )

    def test_end_to_end_model_fit_error(self):
        # Build the model
        model = SegmentAnythingModel(
            backbone=self.image_encoder,
            prompt_encoder=self.prompt_encoder,
            mask_decoder=self.mask_decoder,
        )

        mask_prompts = self.get_prompts(1)
        inputs = {
            "images": np.ones((1, 1024, 1024, 3)),
        }
        inputs.update(mask_prompts)

        # Compile the model
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Check that calling fit raises a NotImplementedError.
        with self.assertRaises(
            NotImplementedError, msg=r"only supports inference"
        ):
            model.fit(inputs)
