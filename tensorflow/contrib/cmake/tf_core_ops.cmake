set(tf_op_lib_names
    "array_ops"
    "candidate_sampling_ops"
    "control_flow_ops"
    "ctc_ops"
    "data_flow_ops"
    "functional_ops"
    "image_ops"
    "io_ops"
    "linalg_ops"
    "logging_ops"
    "math_ops"
    "nn_ops"
    "no_op"
    "parsing_ops"
    "random_ops"
    "resource_variable_ops"
    "script_ops"
    "sdca_ops"
    "sendrecv_ops"
    "sparse_ops"
    "state_ops"
    "string_ops"
    "training_ops"
)

foreach(tf_op_lib_name ${tf_op_lib_names})
    ########################################################
    # tf_${tf_op_lib_name} library
    ########################################################
    file(GLOB tf_${tf_op_lib_name}_srcs
        "${tensorflow_SOURCE_DIR}/tensorflow/core/ops/${tf_op_lib_name}.cc"
    )

    add_library(tf_${tf_op_lib_name} OBJECT ${tf_${tf_op_lib_name}_srcs})

    add_dependencies(tf_${tf_op_lib_name} tf_core_framework)
endforeach()

function(GENERATE_CONTRIB_OP_LIBRARY op_lib_name cc_srcs)
    add_library(tf_contrib_${op_lib_name}_ops OBJECT ${cc_srcs})
    add_dependencies(tf_contrib_${op_lib_name}_ops tf_core_framework)
endfunction()

GENERATE_CONTRIB_OP_LIBRARY(cudnn_rnn "${tensorflow_SOURCE_DIR}/tensorflow/contrib/cudnn_rnn/ops/cudnn_rnn_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(factorization_clustering "${tensorflow_SOURCE_DIR}/tensorflow/contrib/factorization/ops/clustering_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(factorization_factorization "${tensorflow_SOURCE_DIR}/tensorflow/contrib/factorization/ops/factorization_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(framework_variable "${tensorflow_SOURCE_DIR}/tensorflow/contrib/framework/ops/variable_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(metrics_set "${tensorflow_SOURCE_DIR}/tensorflow/contrib/metrics/ops/set_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(word2vec "${tensorflow_SOURCE_DIR}/tensorflow/models/embedding/word2vec_ops.cc")

########################################################
# tf_user_ops library
########################################################
file(GLOB_RECURSE tf_user_ops_srcs
    "${tensorflow_SOURCE_DIR}/tensorflow/core/user_ops/*.cc"
)

add_library(tf_user_ops OBJECT ${tf_user_ops_srcs})

add_dependencies(tf_user_ops tf_core_framework)

########################################################
# tf_core_ops library
########################################################
file(GLOB_RECURSE tf_core_ops_srcs
    "${tensorflow_SOURCE_DIR}/tensorflow/core/ops/*.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/ops/*.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/user_ops/*.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/user_ops/*.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/models/embedding/word2vec_ops.cc"
)

file(GLOB_RECURSE tf_core_ops_exclude_srcs
    "${tensorflow_SOURCE_DIR}/tensorflow/core/ops/*test*.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/ops/*test*.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/ops/*main.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/user_ops/*test*.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/user_ops/*test*.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/user_ops/*main.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/user_ops/*.cu.cc"
)

list(REMOVE_ITEM tf_core_ops_srcs ${tf_core_ops_exclude_srcs}) 

add_library(tf_core_ops OBJECT ${tf_core_ops_srcs})

add_dependencies(tf_core_ops tf_core_cpu)
InstallTFHeaders(tf_core_ops_srcs ${tensorflow_SOURCE_DIR} include)
