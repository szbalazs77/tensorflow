########################################################
# tf_cc_framework library
########################################################
set(tf_cc_framework_srcs
    "${tensorflow_SOURCE_DIR}/tensorflow/cc/framework/ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/cc/framework/ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/cc/framework/scope.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/cc/framework/scope.cc"
)

add_library(tf_cc_framework OBJECT ${tf_cc_framework_srcs})

add_dependencies(tf_cc_framework tf_core_framework)

########################################################
# tf_cc_op_gen_main library
########################################################
set(tf_cc_op_gen_main_srcs
    "${tensorflow_SOURCE_DIR}/tensorflow/cc/framework/cc_op_gen.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/cc/framework/cc_op_gen_main.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/cc/framework/cc_op_gen.h"
)

add_library(tf_cc_op_gen_main OBJECT ${tf_cc_op_gen_main_srcs})

add_dependencies(tf_cc_op_gen_main tf_core_framework)

########################################################
# tf_gen_op_wrapper_cc executables
########################################################

# create directory for ops generated files
set(cc_ops_target_dir ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/cc/ops)

file(MAKE_DIRECTORY ${cc_ops_target_dir})

set(tf_cc_ops_generated_files)
set(tf_cc_ops_generators)

set(tf_cc_op_lib_names
    ${tf_op_lib_names}
    "user_ops"
)
foreach(tf_cc_op_lib_name ${tf_cc_op_lib_names})
    # Using <TARGET_OBJECTS:...> to work around an issue where no ops were
    # registered (static initializers dropped by the linker because the ops
    # are not used explicitly in the *_gen_cc executables).
    add_executable(${tf_cc_op_lib_name}_gen_cc
        $<TARGET_OBJECTS:tf_cc_op_gen_main>
        $<TARGET_OBJECTS:tf_${tf_cc_op_lib_name}>
        $<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_core_framework>
    )

    list(APPEND tf_cc_ops_generators ${tf_cc_op_lib_name}_gen_cc)

    target_link_libraries(${tf_cc_op_lib_name}_gen_cc PRIVATE
        tf_protos_cc
        ${tensorflow_EXTERNAL_LIBRARIES}
    )

    set(cc_ops_include_internal 0)
    if(${tf_cc_op_lib_name} STREQUAL "sendrecv_ops")
        set(cc_ops_include_internal 1)
    endif()

    add_custom_command(
        OUTPUT ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h
               ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc
        COMMAND ${tf_cc_op_lib_name}_gen_cc ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc ${cc_ops_include_internal}
        DEPENDS ${tf_cc_op_lib_name}_gen_cc
    )
    
    list(APPEND tf_cc_ops_generated_files ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h)
    list(APPEND tf_cc_ops_generated_files ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc)
endforeach()


########################################################
# tf_cc_ops library
########################################################
add_library(tf_cc_ops OBJECT
    ${tf_cc_ops_generated_files}
    "${tensorflow_SOURCE_DIR}/tensorflow/cc/ops/const_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/cc/ops/const_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/cc/ops/standard_ops.h"
)

InstallTFHeaders(tf_cc_ops_generated_files ${CMAKE_CURRENT_BINARY_DIR} include)
install(TARGETS ${tf_cc_ops_generators} RUNTIME DESTINATION bin/tensorflow)
