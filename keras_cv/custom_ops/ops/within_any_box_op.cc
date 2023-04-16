/* Copyright 2023 The KerasCV Authors. All Rights Reserved.

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

REGISTER_OP("KcvWithinAnyBox")
    .Input("points: float")
    .Input("boxes: float")
    .Output("within_any_box: bool")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0)}));
      return tensorflow::Status();
    });
