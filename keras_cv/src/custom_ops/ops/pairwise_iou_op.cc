/* Copyright 2022 The KerasCV Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("KcvPairwiseIou3D")
    .Input("boxes_a: float")
    .Input("boxes_b: float")
    .Output("iou: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(
          0, c->MakeShape({c->Dim(c->input(0), 0), c->Dim(c->input(1), 0)}));
      return tensorflow::Status();
    })
    .Doc(R"doc(
Calculate pairwise IoUs between two set of 3D bboxes. Every bbox is represented
as [center_x, center_y, center_z, dim_x, dim_y, dim_z, heading].
boxes_a: A tensor of shape [num_boxes_a, 7]
boxes_b: A tensor of shape [num_boxes_b, 7]
)doc");
