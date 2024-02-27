package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

cc_library(
    name = "box_util",
    srcs = ["box_util.cc"],
    hdrs = ["box_util.h"],
    copts = select({
        ":windows": [
            "/DEIGEN_STRONG_INLINE=inline",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
        ],
    }),
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_binary(
    name = "_keras_cv_custom_ops.so",
    srcs = [
        "kernels/pairwise_iou_kernel.cc",
        "kernels/within_any_box_op.cc",
        "kernels/withinbox_op.cc",
        "ops/pairwise_iou_op.cc",
        "ops/within_any_box_op.cc",
        "ops/withinbox_op.cc",
    ],
    copts = select({
        ":windows": [
            "/DEIGEN_STRONG_INLINE=inline",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
        ],
    }),
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        ":box_util",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
