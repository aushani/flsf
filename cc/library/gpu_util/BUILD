cc_library(
    name = "gpu_util_cuda",
    srcs = glob(['*.cu.cc']),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    copts = ["-x",  "cuda", "-std=c++11"],
    deps = [
            "//tools:cuda",
            "@boost//:boost_headers",
           ],
)

cc_library(
    name = "gpu_util",
    srcs = glob(['*.cc'], exclude=["*.cu.cc"]),
    hdrs = glob(["*.h"], exclude=["*.cu.h"]),
    visibility = ["//visibility:public"],
    deps = [
            ":gpu_util_cuda",
           ],
)
