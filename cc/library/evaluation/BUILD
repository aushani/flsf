cc_library(
    name = "evaluation",
    srcs = glob(['*.cc']),
    hdrs = glob(['*.h']),
    visibility = ["//visibility:public"],
    deps = [
             "//library/kitti",
             "//library/flow",
             "@boost//:boost_headers",
             "@boost//:boost_filesystem",
             "@eigen//:eigen",
           ]
)
