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

#define EIGEN_USE_THREADS

#include "keras_cv/custom_ops/box_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace kerascv {

class WithinBoxOp : public OpKernel {
 public:
  explicit WithinBoxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& points = ctx->input(0);
    const Tensor& boxes = ctx->input(1);
    const int num_points = points.dim_size(0);
    const int num_boxes = boxes.dim_size(0);
    Tensor* box_indices = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("box_indices", TensorShape({num_points}),
                                  &box_indices));
    auto boxes_indices_t = box_indices->flat<int>();
    for (auto i = 0; i < num_points; ++i) boxes_indices_t(i) = -1;

    std::vector<box::Upright3DBox> boxes_vec = box::ParseBoxesFromTensor(boxes);
    std::vector<box::Vertex> points_vec = box::ParseVerticesFromTensor(points);
    std::vector<int> p_indices_x(num_points);
    // index x range [0, num_points)
    std::iota(p_indices_x.begin(), p_indices_x.end(), 0);
    // index y range [0, num_points)
    std::vector<int> p_indices_y(p_indices_x);

    // sort, return sorted value and indices
    std::sort(p_indices_x.begin(), p_indices_x.end(),
              [&points_vec](const int& a, const int& b) -> bool {
                return points_vec[a].x < points_vec[b].x;
              });
    std::sort(p_indices_y.begin(), p_indices_y.end(),
              [&points_vec](const int& a, const int& b) -> bool {
                return points_vec[a].y < points_vec[b].y;
              });
    std::vector<double> sorted_points_x;
    sorted_points_x.reserve(num_points);
    std::vector<double> sorted_points_y;
    sorted_points_y.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
      sorted_points_x.emplace_back(points_vec[p_indices_x[i]].x);
      sorted_points_y.emplace_back(points_vec[p_indices_y[i]].y);
    }

    // for each box, find all point indices whose x values are within box
    // boundaries when the box is rotated, the box boundary is the minimum and
    // maximum x for all vertices
    std::vector<int> points_x_min =
        box::GetMinXIndexFromBoxes(boxes_vec, sorted_points_x);
    std::vector<int> points_x_max =
        box::GetMaxXIndexFromBoxes(boxes_vec, sorted_points_x);
    std::vector<std::unordered_set<int>> points_x_indices(num_boxes);
    auto set_fn_x = [&points_x_min, &points_x_max, &p_indices_x,
                     &points_x_indices](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        std::unordered_set<int> p_set;
        int p_start = points_x_min[idx];
        int p_end = points_x_max[idx];
        for (auto p_idx = p_start; p_idx <= p_end; ++p_idx) {
          p_set.insert(p_indices_x[p_idx]);
        }
        points_x_indices[idx] = p_set;
      }
    };
    const CPUDevice& device = ctx->eigen_device<CPUDevice>();
    const Eigen::TensorOpCost cost(num_points, num_boxes, 3);
    device.parallelFor(num_boxes, cost, set_fn_x);

    // for each box, find all point indices whose y values are within box
    // boundaries when the box is rotated, the box boundary is the minimum and
    // maximum x for all vertices
    std::vector<int> points_y_min =
        box::GetMinYIndexFromBoxes(boxes_vec, sorted_points_y);
    std::vector<int> points_y_max =
        box::GetMaxYIndexFromBoxes(boxes_vec, sorted_points_y);
    std::vector<std::unordered_set<int>> points_y_indices(num_boxes);
    auto set_fn_y = [&points_y_min, &points_y_max, &p_indices_y,
                     &points_y_indices](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        std::unordered_set<int> p_set;
        int p_start = points_y_min[idx];
        int p_end = points_y_max[idx];
        for (auto p_idx = p_start; p_idx <= p_end; ++p_idx) {
          p_set.insert(p_indices_y[p_idx]);
        }
        points_y_indices[idx] = p_set;
      }
    };
    device.parallelFor(num_boxes, cost, set_fn_y);

    // for the intersection of x indices set and y indices set, check if
    // those points are within the box
    auto within_fn = [&points_x_indices, &points_y_indices, &boxes_vec,
                      &points_vec,
                      &boxes_indices_t](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        std::unordered_set<int>& set_a = points_x_indices[idx];
        std::unordered_set<int>& set_b = points_y_indices[idx];
        std::unordered_set<int> p_set;
        for (auto val : set_a) {
          if (set_b.find(val) != set_b.end()) {
            p_set.insert(val);
          }
        }
        box::Upright3DBox& box = boxes_vec[idx];
        for (auto p_idx : p_set) {
          box::Vertex& point = points_vec[p_idx];
          if (box.WithinBox3D(point)) {
            boxes_indices_t(p_idx) = idx;
          }
        }
      }
    };
    device.parallelFor(num_boxes, cost, within_fn);
  }
};

REGISTER_KERNEL_BUILDER(Name("KcvWithinBox").Device(DEVICE_CPU), WithinBoxOp);

}  // namespace kerascv
}  // namespace tensorflow
