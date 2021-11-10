package(
    default_visibility = [
        "//third_party/py/keras_cv:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "coco",
    srcs = [
    ],
    srcs_version = "PY3",
    deps = [
        ":iou",
        ":recall",
    ],
)

py_library(
    name = "bbox",
    srcs = ["bbox.py"],
    srcs_version = "PY3",
)

py_library(
    name = "recall",
    srcs = [
        "recall.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":bbox",
        ":iou",
        "//third_party/py/tensorflow:tensorflow",
    ],
)

py_library(
    name = "iou",
    srcs = [
        "iou.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":bbox",
        "//third_party/py/tensorflow:tensorflow",
    ],
)

py_test(
    name = "iou_test",
    srcs = [
        "iou_test.py",
    ],
    deps = [
        "//third_party/py/keras_cv/metrics/coco:iou",
        "//third_party/py/tensorflow:tensorflow",
    ],
)

py_test(
    name = "recall_test",
    srcs = [
        "recall_test.py",
    ],
    deps = [
        ":recall",
        "//third_party/py/tensorflow:tensorflow",
    ],
)
