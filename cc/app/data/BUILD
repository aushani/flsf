cc_binary(
    name = "extract-samples",
    srcs = glob(["*.cc", "*.h"]),
    deps = [
            "//library/kitti",
            "//library/params",
            "//library/ray_tracing",
            "//library/timer",
            "@boost//:boost_headers",
            "@boost//:boost_filesystem",
            "@boost//:boost_programoptions",
           ],
)
