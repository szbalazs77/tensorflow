########################################################
# tf_stream_executor library
########################################################
file(GLOB tf_stream_executor_srcs
    "${tensorflow_SOURCE_DIR}/tensorflow/stream_executor/*.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/stream_executor/*.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/stream_executor/lib/*.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/stream_executor/lib/*.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/stream_executor/platform/*.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/stream_executor/platform/default/*.h"
)

if (tensorflow_ENABLE_GPU)    
    file(GLOB tf_stream_executor_gpu_srcs
        "${tensorflow_SOURCE_DIR}/tensorflow/stream_executor/cuda/*.cc"
    )
    list(APPEND tf_stream_executor_srcs ${tf_stream_executor_gpu_srcs})
endif()    

add_library(tf_stream_executor OBJECT ${tf_stream_executor_srcs})

add_dependencies(tf_stream_executor
    tf_core_lib
)
