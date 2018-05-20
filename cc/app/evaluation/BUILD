cc_binary(
    name = "evaluate",
    srcs = glob(["*.cc", "*.h"]),
    deps = [
            "//library/evaluation",
            "//library/kitti",
            "//library/ray_tracing",
            "//library/flow",
            "//library/timer",
            "//library/viewer",
            "@boost//:boost_headers",
            "@boost//:boost_filesystem",
            "@boost//:boost_programoptions",
           ],
)
