cc_library(
    name = "flow_cuda",
    srcs = glob(['*.cu.cc']),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    copts = ["-x",  "cuda", "-std=c++11"],
    deps = [
            "//library/gpu_util",
            "//library/kitti",
            "//library/ray_tracing",
            "//library/timer",
            "//library/tf",
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
           ],
)
