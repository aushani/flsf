load("//tools/build_rules:qt.bzl", "qt4_cc_library")

qt4_cc_library(
    name = "viewerwindow-lib",
    src = "viewer_window.cc",
    hdr = "viewer_window.h",
    normal_hdrs = [
                    "pick_handler.h",
                    "viewer_widget.h",
                    "terrain_trackpad_manipulator.h",
                  ],
    linkopts = ["-lpthread"],
    deps = [
            "@qt4//:qt4_core",
            "@qt4//:qt4_gui",
            "@qt4//:qt4_opengl",
            "@qt4//:qt4_core_prefixstripped",
            "@qt4//:qt4_gui_prefixstripped",
            "@qt4//:qt4_opengl_prefixstripped",
            "//library/util",
           ],
)

cc_library (
    name = "viewer",
    srcs = [
            "pick_handler.cc", "pick_handler.h",
            "key_handler.cc", "key_handler.h",
            "terrain_trackpad_manipulator.cc", "terrain_trackpad_manipulator.h",
            "viewer.cc", "viewer.h",
            "viewer_widget.cc", "viewer_widget.h",
            "viewer_window.cc", "viewer_window.h"
            ],
    visibility = ["//visibility:public"],
    deps = [
            ":viewerwindow-lib",
            "@openscenegraph//:openscenegraph",
            "//library/util",
           ],
)
