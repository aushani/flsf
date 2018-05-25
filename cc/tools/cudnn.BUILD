cc_library(
  name = "cudnn",
  hdrs = glob(["include/cudnn.h"]),
  srcs = glob(["lib/x86_64-linux-gnu/libcudnn.so"]),
  visibility = ["//visibility:public"],
  strip_include_prefix = "include",
)
