cc_binary(
    name = "viewer",
    srcs = glob(["*.cc", "*.h"]),
    deps = [
            "//library/kitti",
            "//library/kitti/nodes",
            "//library/params",
            "//library/ray_tracing",
            "//library/ray_tracing/nodes",
            "//library/timer",
            "//library/viewer",
            "//library/osg_nodes",
            "@openscenegraph//:openscenegraph",
            "@boost//:boost_filesystem",
           ],
)
