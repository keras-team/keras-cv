licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

py_library(
    name = "keras_cv",
    srcs = glob(["**/*.py"]),
    data = [
      "//keras_cv/custom_ops:_keras_cv_custom_ops.so",
    ]
)
