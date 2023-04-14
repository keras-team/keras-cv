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

#define EIGEN_USE_THREADS

#include "keras_cv/custom_ops/box_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace kerascv {

class WithinAnyBoxOp : public OpKernel {
 public:
  explicit WithinAnyBoxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& points = ctx->input(0);
    const Tensor& boxes = ctx->input(1);
    const int num_points = points.dim_size(0);
    const int num_boxes = boxes.dim_size(0);
    Tensor* within_any_box = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("within_any_box", TensorShape({num_points}),
                                  &within_any_box));
    auto within_any_box_t = within_any_box->flat<bool>();
    for (auto i = 0; i < num_points; ++i) within_any_box_t(i) = false;

    std::vector<box::Upright3DBox> boxes_vec = box::ParseBoxesFromTensor(boxes);
    std::vector<box::Vertex> points_vec = box::ParseVerticesFromTensor(points);

    auto within_fn = [&boxes_vec, &points_vec, &within_any_box_t](int64_t begin,
                                                                  int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        box::Upright3DBox& box = boxes_vec[idx];
        for (uint64_t p_idx = 0; p_idx < points_vec.size(); ++p_idx) {
          if (within_any_box_t(p_idx)) {
            continue;
          }
          auto point = points_vec[p_idx];
          if (box.WithinBox3D(point)) {
            within_any_box_t(p_idx) = true;
          }
        }
      }
    };
    const CPUDevice& device = ctx->eigen_device<CPUDevice>();
    const Eigen::TensorOpCost cost(num_points, num_boxes, 3);
    device.parallelFor(num_boxes, cost, within_fn);
  }
};

REGISTER_KERNEL_BUILDER(Name("KcvWithinAnyBox").Device(DEVICE_CPU),
                        WithinAnyBoxOp);

}  // namespace kerascv
}  // namespace tensorflow
