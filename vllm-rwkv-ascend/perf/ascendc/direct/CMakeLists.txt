cmake_minimum_required(VERSION 3.16)
project(rwkv7_ascend_direct LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(SOC_VERSION "Ascend910B3" CACHE STRING "Ascend SoC")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH AND DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
  message(FATAL_ERROR "ascendc_kernel_cmake not found under ${ASCEND_CANN_PACKAGE_PATH}")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(rwkv7_shift_mix1_kernel STATIC kernel/rwkv_shift_mix1.cpp)
ascendc_library(rwkv7_shift_mix6_kernel STATIC kernel/rwkv_shift_mix6.cpp)
ascendc_library(rwkv7_state_post_kernel STATIC kernel/rwkv_state_post.cpp)
ascendc_library(rwkv7_k_prep_kernel STATIC kernel/rwkv_k_prep.cpp)
ascendc_library(rwkv7_relu_square_kernel STATIC kernel/rwkv_relu_square.cpp)
ascendc_library(rwkv7_value_mix_kernel STATIC kernel/rwkv_value_mix.cpp)
ascendc_library(rwkv7_head_scaled_add_kernel STATIC kernel/rwkv_head_scaled_add.cpp)
ascendc_library(rwkv7_outer_products_kernel STATIC kernel/rwkv_outer_products.cpp)
ascendc_library(rwkv7_sk_output_kernel STATIC kernel/rwkv_sk_output.cpp)
ascendc_library(rwkv7_normalize_k_kernel STATIC kernel/rwkv_normalize_k.cpp)
ascendc_library(rwkv7_w_pre_kernel STATIC kernel/rwkv_w_pre.cpp)
ascendc_library(
  rwkv7_k_prep_normalize_kernel STATIC kernel/rwkv_k_prep_normalize.cpp)
ascendc_library(
  rwkv7_state_rank1_output_kernel STATIC kernel/rwkv_state_rank1_output.cpp)
ascendc_library(
  rwkv7_groupnorm_sk_kernel STATIC kernel/rwkv_groupnorm_sk.cpp)
ascendc_library(
  rwkv7_lowrank_activate_kernel STATIC kernel/rwkv_lowrank_activate.cpp)
ascendc_library(
  rwkv7_lowrank_post_kernel STATIC kernel/rwkv_lowrank_post.cpp)
ascendc_library(
  rwkv7_recurrence_prep_kernel STATIC kernel/rwkv_recurrence_prep.cpp)
ascendc_library(
  rwkv7_recurrence_state_kernel STATIC kernel/rwkv_recurrence_state.cpp)
ascendc_library(rwkv7_ffn_prep_kernel STATIC kernel/rwkv_ffn_prep.cpp)
ascendc_library(rwkv7_attn_prep_kernel STATIC kernel/rwkv_attn_prep.cpp)
ascendc_library(rwkv7_embedding_kernel STATIC kernel/rwkv_embedding.cpp)
ascendc_library(
  rwkv7_embedding_norm2_kernel STATIC kernel/rwkv_embedding_norm2.cpp)
ascendc_library(
  rwkv7_layer_norm_kernel STATIC kernel/rwkv_layer_norm.cpp)
ascendc_library(
  rwkv7_add_layer_norm_kernel STATIC kernel/rwkv_add_layer_norm.cpp)
ascendc_library(rwkv7_concat2_kernel STATIC kernel/rwkv_concat2.cpp)
add_library(rwkv7_ascend_direct SHARED
  host/rwkv_shift_mix1.cpp host/rwkv_shift_mix6.cpp host/rwkv_state_post.cpp
  host/rwkv_k_prep.cpp host/rwkv_relu_square.cpp host/rwkv_value_mix.cpp
  host/rwkv_head_scaled_add.cpp host/rwkv_outer_products.cpp
  host/rwkv_sk_output.cpp host/rwkv_normalize_k.cpp host/rwkv_w_pre.cpp
  host/rwkv_k_prep_normalize.cpp)
target_link_libraries(rwkv7_ascend_direct PRIVATE
  rwkv7_shift_mix1_kernel rwkv7_shift_mix6_kernel rwkv7_state_post_kernel
  rwkv7_k_prep_kernel
  rwkv7_relu_square_kernel
  rwkv7_value_mix_kernel rwkv7_head_scaled_add_kernel
  rwkv7_outer_products_kernel
  rwkv7_sk_output_kernel
  rwkv7_normalize_k_kernel
  rwkv7_w_pre_kernel
  rwkv7_k_prep_normalize_kernel
  rwkv7_state_rank1_output_kernel
  rwkv7_groupnorm_sk_kernel
  rwkv7_lowrank_activate_kernel
  rwkv7_lowrank_post_kernel
  rwkv7_recurrence_prep_kernel
  rwkv7_recurrence_state_kernel
  rwkv7_ffn_prep_kernel
  rwkv7_attn_prep_kernel
  rwkv7_embedding_kernel
  rwkv7_embedding_norm2_kernel
  rwkv7_layer_norm_kernel
  rwkv7_add_layer_norm_kernel
  rwkv7_concat2_kernel
  torch_npu ascendcl register platform ascendalog dl)
target_link_directories(rwkv7_ascend_direct PRIVATE
  ${TORCH_PATH}/lib ${TORCH_NPU_PATH}/lib)
target_include_directories(rwkv7_ascend_direct PRIVATE
  ${TORCH_NPU_PATH}/include
  ${TORCH_NPU_PATH}/include/third_party/acl/inc
  ${TORCH_PATH}/include
  ${TORCH_PATH}/include/torch/csrc/api/include)
