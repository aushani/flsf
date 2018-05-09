cc_library(
    name = "tf_cuda",
    srcs = glob(['*.cu.cc']),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    copts = ["-x",  "cuda", "-std=c++11"],
    deps = [
            "//library/timer",
            "//library/gpu_util",
            "//library/params",
            "//library/util",
            "//library/ray_tracing:ray_tracing_cuda",
            "//tools:cuda",
            "@boost//:boost_headers",
            "@boost//:boost_filesystem",
           ],
)

cc_library(
    name = "tf",
    srcs = glob(['*.cc',], exclude=["*.cu.cc"]),
    hdrs = glob(["*.h", '*.hpp'], exclude=["*.cu.h"]),
    visibility = ["//visibility:public"],
    deps = [
            ":tf_cuda",
            "@boost//:boost_filesystem",
           ],
)
