########################################################
# tf_core_direct_session library
########################################################
file(GLOB tf_core_direct_session_srcs
   "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/direct_session.cc"
   "${tensorflow_SOURCE_DIR}/tensorflow/core/common_runtime/direct_session.h"
   "${tensorflow_SOURCE_DIR}/tensorflow/core/debug/*.h"
   "${tensorflow_SOURCE_DIR}/tensorflow/core/debug/*.cc"
)

file(GLOB_RECURSE tf_core_direct_session_test_srcs
    "${tensorflow_SOURCE_DIR}/tensorflow/core/debug/*test*.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/debug/*test*.cc"
)

list(REMOVE_ITEM tf_core_direct_session_srcs ${tf_core_direct_session_test_srcs})

add_library(tf_core_direct_session OBJECT ${tf_core_direct_session_srcs})

add_dependencies(tf_core_direct_session tf_core_cpu)

InstallTFHeaders(tf_core_direct_session_srcs ${tensorflow_SOURCE_DIR} include)
