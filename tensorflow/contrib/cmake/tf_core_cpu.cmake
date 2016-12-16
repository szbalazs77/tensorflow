########################################################
# tf_core_cpu library
########################################################
file(GLOB_RECURSE tf_core_cpu_srcs
    "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/*.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/*.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/graph/*.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/graph/*.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/public/*.h"
)

file(GLOB_RECURSE tf_core_cpu_exclude_srcs
    "${tensorflow_SOURCE_DIR}/tensorflow/core/*test*.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/*test*.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/*main.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/gpu/*.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/gpu_device_factory.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/direct_session.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/direct_session.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/session.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/session_factory.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/session_options.cc"
)
list(REMOVE_ITEM tf_core_cpu_srcs ${tf_core_cpu_exclude_srcs}) 

# We need to include stubs for the GPU tracer, which are in the exclude glob.
list(APPEND tf_core_cpu_srcs
     "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/gpu/gpu_tracer.cc"
     "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/gpu/gpu_tracer.h"
)

if (tensorflow_ENABLE_GPU)
  file(GLOB_RECURSE tf_core_gpu_srcs
    "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/gpu/*.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/platform/default/gpu/cupti_wrapper.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/gpu_device_factory.cc"
  )
  file(GLOB_RECURSE tf_core_gpu_exclude_srcs
     "${tensorflow_SOURCE_DIR}/tensorflow/core/*test*.cc"
     "${tensorflow_SOURCE_DIR}/tensorflow/core/*test*.cc"
  )
  list(REMOVE_ITEM tf_core_gpu_srcs ${tf_core_gpu_exclude_srcs})
  list(APPEND tf_core_cpu_srcs ${tf_core_gpu_srcs})
endif()

add_library(tf_core_cpu OBJECT ${tf_core_cpu_srcs})
add_dependencies(tf_core_cpu tf_core_framework)

InstallTFHeaders(tf_core_cpu_srcs ${tensorflow_SOURCE_DIR} include)
