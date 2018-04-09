cc_binary(
    name = "viewer",
    srcs = glob(["*.cc", "*.h"]),
    deps = [
            "//library/kitti",
            "//library/kitti/nodes",
            "//library/ray_tracing",
            "//library/ray_tracing/nodes",
            "//library/timer",
            #"//library/gpu_util",
            "//library/viewer",
            "@openscenegraph//:openscenegraph",
           ],
)
