# tensorflow is a shared library containing all of the
# TensorFlow runtime and the standard ops and kernels.
if (tensorflow_BUILD_ALL_KERNELS)
  set(tf_kernel_objects $<TARGET_OBJECTS:tf_core_kernels>)
else ()
  set(tf_kernel_objects
    $<TARGET_OBJECTS:tf_core_kernels_android>
    $<TARGET_OBJECTS:tf_core_kernels>)
endif ()

add_library(tensorflow STATIC
    $<TARGET_OBJECTS:tf_c>
    $<TARGET_OBJECTS:tf_cc>
    $<TARGET_OBJECTS:tf_cc_framework>
    $<TARGET_OBJECTS:tf_cc_ops>
    $<TARGET_OBJECTS:tf_cc_while_loop>
    $<TARGET_OBJECTS:tf_core_lib>
    $<TARGET_OBJECTS:tf_core_cpu>
    $<TARGET_OBJECTS:tf_core_framework>
    $<TARGET_OBJECTS:tf_core_ops>
    $<TARGET_OBJECTS:tf_core_direct_session>
    $<TARGET_OBJECTS:tf_tools_transform_graph_lib>
    $<$<BOOL:${tensorflow_ENABLE_GRPC_SUPPORT}>:$<TARGET_OBJECTS:tf_core_distributed_runtime>>
    ${tf_kernel_objects}
    $<$<AND:$<BOOL:${tensorflow_ENABLE_GPU}>,$<BOOL:${WIN32}>>:$<TARGET_OBJECTS:tf_core_kernels_cpu_only>>
    $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
    ${tensorflow_deffile}
)
#set_target_properties(tensorflow PROPERTIES VERSION ${tensorflow_VERSION} SOVERSION ${tensorflow_VERSION_MAJOR})

add_dependencies(tensorflow tf_protos_cc)
set(tensorflow_dependencies
    $<TARGET_FILE:tensorflow>
    $<TARGET_FILE:tf_protos_cc>
)

target_link_libraries(tensorflow PRIVATE
    ${tf_core_gpu_kernels_lib}
    ${tensorflow_EXTERNAL_LIBRARIES}
    tf_protos_cc
)	

# There is a bug in GCC 5 resulting in undefined reference to a __cpu_model function when
# linking to the tensorflow library. Adding the following libraries fixes it.
# See issue on github: https://github.com/tensorflow/tensorflow/issues/9593
if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    target_link_libraries(tensorflow PRIVATE gcc_s gcc)
endif()

#add_dependencies(tensorflow tensorflow_static)

install(TARGETS tensorflow ARCHIVE DESTINATION lib)
set(tensorflow_LIB_REFS "-ltensorflow")
set(tensorflow_c_hdrs "${tensorflow_SOURCE_DIR}/tensorflow/c/c_api.h")
InstallTFHeaders(tensorflow_c_hdrs ${tensorflow_SOURCE_DIR} include)

install(DIRECTORY third_party/eigen3 DESTINATION include/tensorflow/third_party)

configure_file(${tensorflow_SOURCE_DIR}/tensorflow.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/tensorflow.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tensorflow.pc DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
