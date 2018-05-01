cc_library(
    name = "flow_util",
    hdrs = [
             "flow_image.h",
             "classification_map.h",
             "filter_map.h",
             "distance_map.h",
           ],
    srcs = [
             "flow_image.cc",
             "classification_map.cc",
             "filter_map.cc",
             "distance_map.cc",
           ],
    deps = [
             "//library/kitti",
           ]
)

cc_library(
    name = "flow_cuda",
    srcs = glob(['*.cu.cc']),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    copts = ["-x",  "cuda", "-std=c++11"],
    deps = [
            ":flow_util",
            "//library/gpu_util",
            "//library/kitti",
            "//library/ray_tracing",
            "//library/timer",
            "//library/tf",
            "//library/params",
            "//tools:cuda",
            "@boost//:boost_headers",
            "@boost//:boost_filesystem",
           ],
)

cc_library(
    name = "flow",
    srcs = glob(['*.cc',], exclude=["*.cu.cc"]),
    hdrs = glob(["*.h", '*.hpp'], exclude=["*.cu.h"]),
    visibility = ["//visibility:public"],
    deps = [
            ":flow_cuda",
            ":flow_util",
           ],
)
