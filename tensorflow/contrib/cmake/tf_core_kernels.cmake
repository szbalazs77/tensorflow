########################################################
# tf_core_kernels library
########################################################

if(tensorflow_BUILD_ALL_KERNELS)
  file(GLOB_RECURSE tf_core_kernels_srcs
     "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*.h"
     "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*.cc"
  )
else(tensorflow_BUILD_ALL_KERNELS)
  # Use the android core kernels as the subset to include.
  set(tf_core_kernels_android_srcs
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/avgpooling_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/bounds_check.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_ops_common.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_ops_gradients.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_activations.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_attention.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_backward_cuboid_convolutions.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_backward_spatial_convolutions.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_cuboid_convolution.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_patch_3d.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_pooling.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_softmax.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_spatial_convolutions.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fifo_queue.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/maxpooling_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/ops_util.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/ops_util.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/padding_fifo_queue.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/pooling_ops_common.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/pooling_ops_common.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/queue_base.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/queue_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/typed_queue.h"
  )
  add_library(tf_core_kernels_android OBJECT ${tf_core_kernels_android_srcs})
  add_dependencies(tf_core_kernels_android tf_core_framework)
  InstallTFHeaders(tf_core_kernels_android_srcs ${tensorflow_SOURCE_DIR} include)
  set(tf_android_tensorflow_kernels
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/aggregate_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/aggregate_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/aggregate_ops_cpu.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/assign_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/bias_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/bias_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/bounds_check.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_bfloat.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_bool.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_complex128.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_complex64.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_double.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_float.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_half.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_int16.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_int32.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_int64.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_int8.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_uint16.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cast_op_impl_uint8.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/concat_lib.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/concat_lib_cpu.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/concat_lib_cpu.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/concat_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/constant_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/constant_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_ops_common.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_ops_common.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_ops_gradients.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/dense_update_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/dense_update_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/example_parsing_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fill_functor.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fill_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/function_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/gather_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/gather_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/identity_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/identity_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/immutable_constant_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/immutable_constant_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/matmul_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/matmul_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/no_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/no_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/ops_util.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/pack_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/pooling_ops_common.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reshape_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reshape_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reverse_sequence_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reverse_sequence_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/sendrecv_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/sendrecv_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/sequence_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/shape_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_1.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_2.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_3.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_4.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_5.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_6.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softmax_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softmax_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softmax_op_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/split_lib.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/split_lib_cpu.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/split_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/split_v_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/strided_slice_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/strided_slice_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/strided_slice_op_impl.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/strided_slice_op_inst_0.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/strided_slice_op_inst_1.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/strided_slice_op_inst_2.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/strided_slice_op_inst_3.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/strided_slice_op_inst_4.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/strided_slice_op_inst_5.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/strided_slice_op_inst_6.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/unpack_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/variable_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/variable_ops.h"
  )
  set(tf_android_extended_ops_headers
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/argmax_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/avgpooling_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/batch_norm_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/control_flow_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_2d.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/depthwise_conv_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fake_quant_ops_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/gemm_functors.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/image_resizer_state.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/maxpooling_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mirror_pad_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mirror_pad_op_cpu_impl.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/pad_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/random_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_common.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/relu_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/relu_op_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/resize_bilinear_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reverse_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/save_restore_tensor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softplus_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softsign_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tensor_array.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops_cpu_impl.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops_impl.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/training_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/transpose_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/transpose_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/where_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/xent_op.h"
  )
  set(tf_android_extended_ops_group1
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/argmax_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/avgpooling_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/batch_norm_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/bcast_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/check_numerics_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/control_flow_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_2d.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_grad_filter_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_grad_input_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_grad_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_grad_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_ops_fused.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_ops_using_gemm.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_abs.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_add_1.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_add_2.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_div.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_equal_to_1.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_equal_to_2.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_exp.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_floor.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_greater.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_greater_equal.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_isfinite.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_less.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_log.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_logical_and.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_maximum.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_minimum.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_mul_1.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_mul_2.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_neg.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_reciprocal.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_rsqrt.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_select.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_sigmoid.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_sqrt.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_square.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_squared_difference.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_sub.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_tanh.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/deep_conv2d.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/deep_conv2d.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/depthwise_conv_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/dynamic_partition_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fake_quant_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fifo_queue.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fused_batch_norm_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/winograd_transform.h"
  )
  set(tf_android_extended_ops_group2
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/ctc_decoder_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/dynamic_stitch_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/in_topk_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/logging_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/lrn_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/maxpooling_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mirror_pad_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mirror_pad_op_cpu_impl_1.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mirror_pad_op_cpu_impl_2.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mirror_pad_op_cpu_impl_3.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mirror_pad_op_cpu_impl_4.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mirror_pad_op_cpu_impl_5.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/pad_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/padding_fifo_queue.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/padding_fifo_queue_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/queue_base.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/queue_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/random_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_common.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_max.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_mean.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_min.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_prod.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_sum.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/relu_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/resize_bicubic_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/resize_bilinear_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/resize_area_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/resize_nearest_neighbor_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/restore_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reverse_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/save_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/save_restore_tensor.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/save_restore_v2_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/session_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softplus_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softsign_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/sparse_to_dense_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/stack_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/summary_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tensor_array.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tensor_array_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops_cpu_impl_1.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops_cpu_impl_2.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops_cpu_impl_3.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops_cpu_impl_4.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops_cpu_impl_5.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops_cpu_impl_6.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/topk_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/training_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/transpose_functor_cpu.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/transpose_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/where_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/xent_op.cc"
  )
  set(tf_android_quantized_ops
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/dequantize_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/meta_support.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/meta_support.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantization_utils.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantization_utils.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantize_down_and_shrink_range.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantize_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_activation_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_batch_norm_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_bias_add_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_concat_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_conv_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_matmul_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_pooling_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_reshape_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reference_gemm.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/requantization_range_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/requantize.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reshape_op.h"
  )
  set(tf_core_kernels_srcs
    ${tf_android_tensorflow_kernels}
    ${tf_android_extended_ops_headers}
    ${tf_android_extended_ops_group1}
    ${tf_android_extended_ops_group2}
    ${tf_android_quantized_ops}
  )
  list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_android_srcs})
endif(tensorflow_BUILD_ALL_KERNELS)

if(tensorflow_BUILD_CONTRIB_KERNELS)
  set(tf_contrib_kernels_srcs
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/factorization/kernels/clustering_ops.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/factorization/kernels/wals_solver_ops.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/factorization/ops/clustering_ops.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/factorization/ops/factorization_ops.cc"
      #"${tensorflow_SOURCE_DIR}/tensorflow/contrib/ffmpeg/decode_audio_op.cc"
      #"${tensorflow_SOURCE_DIR}/tensorflow/contrib/ffmpeg/encode_audio_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/layers/kernels/bucketization_kernel.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/layers/kernels/sparse_feature_cross_kernel.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/layers/ops/bucketization_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/layers/ops/sparse_feature_cross_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/metrics/kernels/set_kernels.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/metrics/ops/set_ops.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/rnn/kernels/blas_gemm.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/rnn/kernels/gru_ops.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/rnn/kernels/lstm_ops.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/rnn/ops/gru_ops.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/rnn/ops/lstm_ops.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/core/ops/best_splits_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/core/ops/count_extremely_random_stats_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/core/ops/finished_nodes_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/core/ops/grow_tree_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/core/ops/sample_inputs_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/core/ops/scatter_add_ndim_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/core/ops/topn_ops.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/core/ops/tree_predictions_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/core/ops/tree_utils.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/core/ops/update_fertile_slots_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/data/string_to_float_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/hybrid/core/ops/hard_routing_function_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/hybrid/core/ops/k_feature_gradient_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/hybrid/core/ops/k_feature_routing_function_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/hybrid/core/ops/routing_function_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/hybrid/core/ops/routing_gradient_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/hybrid/core/ops/stochastic_hard_routing_function_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/hybrid/core/ops/stochastic_hard_routing_gradient_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/hybrid/core/ops/unpack_path_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/contrib/tensor_forest/hybrid/core/ops/utils.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/models/embedding/word2vec_kernels.cc" 
    )
  list(APPEND tf_core_kernels_srcs ${tf_contrib_kernels_srcs})
endif(tensorflow_BUILD_CONTRIB_KERNELS)


file(GLOB_RECURSE tf_core_kernels_exclude_srcs
   "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*test*.h"
   "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*test*.cc"
   "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*testutil.h"
   "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*testutil.cc"
   "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*main.cc"
   "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*.cu.cc"
   "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/debug_ops.h"  # stream_executor dependency
   "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/debug_ops.cc"  # stream_executor dependency
)

if (NOT tensorflow_ENABLE_GIF)
    file(GLOB tf_core_kernels_gif_srcs
      "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*gif*")
    list(APPEND tf_core_kernels_exclude_srcs ${tf_core_kernels_gif_srcs})
endif()

if (NOT tensorflow_ENABLE_JPEG)
    file(GLOB tf_core_kernels_jpeg_srcs
      "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*jpeg*")
    list(APPEND tf_core_kernels_exclude_srcs ${tf_core_kernels_jpeg_srcs})
endif()

if (NOT tensorflow_ENABLE_PNG)
    file(GLOB tf_core_kernels_png_srcs
      "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*png*"
      "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*summary_image*")
    list(APPEND tf_core_kernels_exclude_srcs ${tf_core_kernels_png_srcs})
endif()

list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_exclude_srcs})

if(WIN32)
  file(GLOB_RECURSE tf_core_kernels_windows_exclude_srcs
      # not working on windows yet
      "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fact_op.cc"
      "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/meta_support.*"
      "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*quantiz*.h"
      "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*quantiz*.cc"
  )
  list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_windows_exclude_srcs})
endif(WIN32)

file(GLOB_RECURSE tf_core_gpu_kernels_srcs
   "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/*.cu.cc"
   "${tensorflow_SOURCE_DIR}/tensorflow/contrib/rnn/kernels/*.cu.cc"
)

add_library(tf_core_kernels OBJECT ${tf_core_kernels_srcs})
add_dependencies(tf_core_kernels tf_core_cpu)

if(WIN32)
  target_compile_options(tf_core_kernels PRIVATE /MP)
  if (tensorflow_ENABLE_GPU)
    set_source_files_properties(${tf_core_gpu_kernels_srcs} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set(tf_core_gpu_kernels_lib tf_core_gpu_kernels)
    cuda_add_library(${tf_core_gpu_kernels_lib} ${tf_core_gpu_kernels_srcs})
    set_target_properties(${tf_core_gpu_kernels_lib}
                          PROPERTIES DEBUG_POSTFIX ""
                          COMPILE_FLAGS "${TF_REGULAR_CXX_FLAGS}"
    )
    add_dependencies(${tf_core_gpu_kernels_lib} tf_core_cpu)
  endif()
endif()

InstallTFHeaders(tf_core_kernels_srcs ${tensorflow_SOURCE_DIR} include)
