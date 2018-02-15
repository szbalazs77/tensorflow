# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_pooling.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_softmax.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_spatial_convolutions.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/eigen_volume_patch.h"
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
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/dense_update_functor.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/dense_update_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/dense_update_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/example_parsing_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fill_functor.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fill_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/function_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/gather_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/gather_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/identity_n_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/identity_n_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/identity_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/identity_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/immutable_constant_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/immutable_constant_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/matmul_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/matmul_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/no_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/no_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/non_max_suppression_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/non_max_suppression_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/one_hot_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/one_hot_op.h"
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
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/shape_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_1.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_2.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_3.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_4.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_5.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_6.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/slice_op_cpu_impl_7.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softmax_op.cc"
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
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/strided_slice_op_inst_7.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/unpack_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/variable_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/variable_ops.h"
  )
  set(tf_android_extended_ops_headers
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/argmax_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/avgpooling_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/batch_matmul_op_impl.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/batch_norm_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/control_flow_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_2d.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/conv_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/depthtospace_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/depthwise_conv_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fake_quant_ops_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fused_batch_norm_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/gemm_functors.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/image_resizer_state.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/maxpooling_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mfcc.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mfcc_dt.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mfcc_mel_filterbank.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mirror_pad_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mirror_pad_op_cpu_impl.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/pad_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/random_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_common.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/relu_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/relu_op_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/resize_bilinear_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/resize_nearest_neighbor_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reverse_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/save_restore_tensor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softplus_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softsign_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tensor_array.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops_cpu_impl.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops_impl.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/topk_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/training_op_helpers.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/training_ops.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/transpose_functor.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/transpose_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/warn_about_ints.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/where_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/xent_op.h"
  )
  set(tf_android_extended_ops_group1
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/argmax_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/avgpooling_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/batch_matmul_op_real.cc"
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
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/crop_and_resize_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/crop_and_resize_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_abs.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_add_1.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_add_2.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_bitwise_and.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_bitwise_or.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_bitwise_xor.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_div.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_equal_to_1.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_equal_to_2.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_exp.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_floor.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_floor_div.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_floor_mod.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_greater.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_greater_equal.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_invert.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_isfinite.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_less.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_less_equal.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_log.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_logical_and.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_logical_not.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_logical_or.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_maximum.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_minimum.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_mul_1.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_mul_2.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_neg.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_pow.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_reciprocal.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_round.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_rsqrt.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_select.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_sigmoid.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_sign.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_sqrt.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_square.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_squared_difference.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_sub.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/cwise_op_tanh.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/decode_wav_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/deep_conv2d.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/deep_conv2d.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/depthwise_conv_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/dynamic_partition_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/encode_wav_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fake_quant_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fifo_queue.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/fused_batch_norm_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/population_count_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/population_count_op.h"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/winograd_transform.h"
  )
  set(tf_android_extended_ops_group2
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/batchtospace_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/ctc_decoder_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/decode_bmp_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/depthtospace_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/dynamic_stitch_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/in_topk_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/logging_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/lrn_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/maxpooling_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mfcc.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mfcc_dct.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mfcc_mel_filterbank.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/mfcc_op.cc"
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
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_all.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_any.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_common.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_max.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_mean.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_min.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_prod.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reduction_ops_sum.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/relu_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/resize_bilinear_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/resize_nearest_neighbor_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/restore_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/reverse_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/save_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/save_restore_tensor.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/save_restore_v2_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/session_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softplus_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/softsign_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/spacetobatch_functor.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/spacetobatch_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/spacetodepth_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/sparse_to_dense_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/spectrogram.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/spectrogram_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/stack_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/string_join_op.cc"
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
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/tile_ops_cpu_impl_7.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/topk_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/training_op_helpers.cc"
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
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_instance_norm.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_matmul_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_pooling_ops.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_reshape_op.cc"
    "${tensorflow_SOURCE_DIR}/tensorflow/core/kernels/quantized_resize_bilinear_op.cc"
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
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/model_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/prediction_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/quantile_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/split_handler_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/stats_accumulator_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/kernels/training_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/batch_features.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/dropout_utils.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/examples_iterable.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/parallel_for.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/sparse_column_iterable.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/utils/tensor_utils.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/learner/common/partitioners/example_partitioner.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/learner/stochastic/handlers/bias-feature-column-handler.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/learner/stochastic/handlers/categorical-feature-column-handler.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/learner/stochastic/handlers/dense-quantized-feature-column-handler.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/learner/stochastic/handlers/sparse-quantized-feature-column-handler.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/models/multiple_additive_trees.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/lib/trees/decision_tree.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/model_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/prediction_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/quantile_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/split_handler_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/stats_accumulator_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/training_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/cudnn_rnn/kernels/cudnn_rnn_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/cudnn_rnn/ops/cudnn_rnn_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/kernels/clustering_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/kernels/masked_matmul_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/kernels/wals_solver_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/ops/clustering_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/ops/factorization_ops.cc"
      #"${tensorflow_source_dir}/tensorflow/contrib/ffmpeg/decode_audio_op.cc"
      #"${tensorflow_source_dir}/tensorflow/contrib/ffmpeg/encode_audio_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/framework/kernels/zero_initializer_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/framework/ops/variable_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/image/kernels/adjust_hsv_in_yiq_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/image/kernels/bipartite_match_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/image/kernels/image_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/image/kernels/single_image_random_dot_stereograms_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/image/ops/distort_image_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/image/ops/image_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/image/ops/single_image_random_dot_stereograms_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/layers/kernels/sparse_feature_cross_kernel.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/layers/ops/sparse_feature_cross_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/nccl/kernels/nccl_manager.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/nccl/kernels/nccl_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/nccl/ops/nccl_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/kernels/hyperplane_lsh_probes.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/ops/nearest_neighbor_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/resampler/kernels/resampler_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/resampler/ops/resampler_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/blas_gemm.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/gru_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/lstm_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/gru_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/lstm_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/kernels/beam_search_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/ops/beam_search_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/ops/tensor_forest_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/reinterpret_string_to_float_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/scatter_add_ndim_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/kernels/tree_utils.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/hard_routing_function_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/k_feature_gradient_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/k_feature_routing_function_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/routing_function_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/routing_gradient_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/stochastic_hard_routing_function_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/stochastic_hard_routing_gradient_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/unpack_path_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/utils.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/text/kernels/skip_gram_kernels.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/text/ops/skip_gram_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tpu/ops/cross_replica_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tpu/ops/infeed_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tpu/ops/outfeed_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tpu/ops/replication_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tpu/ops/tpu_configuration_ops.cc"
    )
  list(APPEND tf_core_kernels_srcs ${tf_contrib_kernels_srcs})
endif(tensorflow_BUILD_CONTRIB_KERNELS)

if(NOT tensorflow_ENABLE_SSL_SUPPORT)
  # Cloud libraries require boringssl.
  file(GLOB tf_core_kernels_cloud_srcs
      "${tensorflow_source_dir}/tensorflow/contrib/cloud/kernels/*.h"
      "${tensorflow_source_dir}/tensorflow/contrib/cloud/kernels/*.cc"
  )
list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_cloud_srcs})
endif()

file(GLOB_RECURSE tf_core_kernels_exclude_srcs
   "${tensorflow_source_dir}/tensorflow/core/kernels/*test*.h"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*test*.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*testutil.h"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*testutil.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*test_utils.h"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*test_utils.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*main.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*.cu.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/fuzzing/*"
   "${tensorflow_source_dir}/tensorflow/core/kernels/hexagon/*"
   "${tensorflow_source_dir}/tensorflow/core/kernels/remote_fused_graph_rewriter_transform*.cc"
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

if(NOT tensorflow_ENABLE_AUDIO_SUPPORT)
    file(GLOB tf_core_kernels_audio_srcs
       "${tensorflow_source_dir}/tensorflow/core/kernels/*audio*"
       "${tensorflow_source_dir}/tensorflow/core/kernels/*code_wav*"
       "${tensorflow_source_dir}/tensorflow/core/kernels/*mfcc*"
       "${tensorflow_source_dir}/tensorflow/core/kernels/*spectrogram*"
    )
    list(APPEND tf_core_kernels_exclude_srcs ${tf_core_kernels_audio_srcs})
endif()

list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_exclude_srcs})

if(WIN32)
  file(GLOB_RECURSE tf_core_kernels_windows_exclude_srcs
      # not working on windows yet
      "${tensorflow_source_dir}/tensorflow/core/kernels/meta_support.*"
      "${tensorflow_source_dir}/tensorflow/core/kernels/*quantiz*.h"
      "${tensorflow_source_dir}/tensorflow/core/kernels/*quantiz*.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/neon/*"
      # not in core - those are loaded dynamically as dll
      "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/kernels/hyperplane_lsh_probes.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/ops/nearest_neighbor_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/resampler/kernels/resampler_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/blas_gemm.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/gru_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/lstm_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/gru_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/lstm_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/kernels/beam_search_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/ops/beam_search_ops.cc"
      # temporarily disable nccl (nccl itself needs to be ported to windows first)
      "${tensorflow_source_dir}/tensorflow/contrib/nccl/kernels/nccl_manager.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/nccl/kernels/nccl_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/nccl/ops/nccl_ops.cc"
  )
  list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_windows_exclude_srcs})
else(WIN32)
  if(NOT tensorflow_ENABLE_NCCL_SUPPORT)
    file(GLOB_RECURSE tf_core_kernels_nccl_srcs
      "${tensorflow_source_dir}/tensorflow/contrib/nccl/kernels/nccl_manager.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/nccl/kernels/nccl_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/nccl/ops/nccl_ops.cc"
    )
    list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_nccl_srcs})
  endif()
endif(WIN32)

file(GLOB_RECURSE tf_core_gpu_kernels_srcs
    "${tensorflow_source_dir}/tensorflow/core/kernels/*.cu.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/framework/kernels/zero_initializer_op_gpu.cu.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/image/kernels/*.cu.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/*.cu.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/kernels/*.cu.cc"
)

if(WIN32 AND tensorflow_ENABLE_GPU)
  file(GLOB_RECURSE tf_core_kernels_cpu_only_srcs
      # GPU implementation not working on Windows yet.
      "${tensorflow_source_dir}/tensorflow/core/kernels/matrix_diag_op.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/one_hot_op.cc")
  list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_cpu_only_srcs})
  add_library(tf_core_kernels_cpu_only OBJECT ${tf_core_kernels_cpu_only_srcs})
  add_dependencies(tf_core_kernels_cpu_only tf_core_cpu)
  # Undefine GOOGLE_CUDA to avoid registering unsupported GPU kernel symbols.
  get_target_property(target_compile_flags tf_core_kernels_cpu_only COMPILE_FLAGS)
  if(target_compile_flags STREQUAL "target_compile_flags-NOTFOUND")
    set(target_compile_flags "/UGOOGLE_CUDA")
  else()
    set(target_compile_flags "${target_compile_flags} /UGOOGLE_CUDA")
  endif()
  set_target_properties(tf_core_kernels_cpu_only PROPERTIES COMPILE_FLAGS ${target_compile_flags})
endif(WIN32 AND tensorflow_ENABLE_GPU)

add_library(tf_core_kernels OBJECT ${tf_core_kernels_srcs})
add_dependencies(tf_core_kernels tf_core_cpu)

if(WIN32)
  target_compile_options(tf_core_kernels PRIVATE /MP)
endif()
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

InstallTFHeaders(tf_core_kernels_srcs ${tensorflow_SOURCE_DIR} include)
