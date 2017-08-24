# Minimum CMake required
cmake_minimum_required(VERSION 3.4)

# Project
project(tensorflow C CXX)

set(tensorflow_VERSION_MAJOR "1")
set(tensorflow_VERSION_MINOR "0")
set(tensorflow_VERSION_PATCH "1")
set(tensorflow_VERSION "${tensorflow_VERSION_MAJOR}.${tensorflow_VERSION_MINOR}.${tensorflow_VERSION_PATCH}")
execute_process(COMMAND git describe --long --dirty --tags
                WORKING_DIRECTORY ${tensorflow_SOURCE_DIR}
                OUTPUT_VARIABLE tensorflow_VERSION_STRING
                RESULT_VARIABLE _tensorflow_git_result
                ERROR_QUIET
		OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT ${_tensorflow_git_result} EQUAL 0)
  set(tensorflow_VERSION_STRING "unknown")
endif()

# Set C++14 as standard for the whole project
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -Wno-narrowing")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
else()
    set(CMAKE_CXX_STANDARD 14)
endif()


list(INSERT CMAKE_MODULE_PATH 0 "${tensorflow_SOURCE_DIR}/tensorflow/contrib/cmake")

# For compatibility with the original rules that had CMakeLists.txt
# in with the cmake modules, rather than at the top of the tree
set(tensorflow_source_dir "${tensorflow_SOURCE_DIR}")

# Options
option(tensorflow_VERBOSE "Enable for verbose output" OFF)
option(tensorflow_ENABLE_GPU "Enable GPU support" OFF)
option(tensorflow_ENABLE_SSL_SUPPORT "Enable boringssl support" OFF)
option(tensorflow_ENABLE_GRPC_SUPPORT "Enable gRPC support" ON)
option(tensorflow_BUILD_CC_EXAMPLE "Build the C++ tutorial example" ON)
option(tensorflow_BUILD_PYTHON_BINDINGS "Build the Python bindings" ON)
option(tensorflow_BUILD_ALL_KERNELS "Build all OpKernels" ON)
option(tensorflow_BUILD_CONTRIB_KERNELS "Build OpKernels from tensorflow/contrib/..." ON)
option(tensorflow_BUILD_CC_TESTS "Build cc unit tests " OFF)
option(tensorflow_BUILD_PYTHON_TESTS "Build python unit tests " OFF)
option(tensorflow_OPTIMIZE_FOR_NATIVE_ARCH "Enable compiler optimizations for the native processor architecture (if available)" ON)
option(tensorflow_BUILD_DEPENDENCIES "Download and build dependent packages" OFF)
option(tensorflow_BUILD_PROTO_TEXT "Build proto_text tool" ON)
option(tensorflow_ENABLE_GIF "Enable GIF support" OFF)
option(tensorflow_ENABLE_JPEG "Enable JPEG support" OFF)
option(tensorflow_ENABLE_PNG "Enable PNG support" OFF)
option(tensorflow_BUILD_LIBRARIES "Build and install libraries and headers" ON)
option(tensorflow_ENABLE_SLIM_BUILD "Exclude zlib compression" OFF)
option(tensorflow_ENABLE_TYPES_SLIM "Reduce number of supported types" OFF)
option(tensorflow_ENABLE_SELECTIVE_REGISTRATION "Enable selective registration of ops" OFF)

if (tensorflow_ENABLE_SLIM_BUILD)
  if (NOT tensorflow_ENABLE_PNG)
    set(_tensorflow_need_zlib OFF)
  else()
    set(_tensorflow_need_zlib ON)
  endif()
else()
  set(_tensorflow_need_zlib ON)
endif()

if (NOT WIN32)
  # Threads: defines CMAKE_THREAD_LIBS_INIT and adds -pthread compile option
  # for targets that link ${CMAKE_THREAD_LIBS_INIT}.
  find_package (Threads)
endif()

# [CLEANUP] Remove when done
# For debugging
function(SHOW_VARIABLES)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endfunction()

function(InstallTFHeaders HDR_VAR BASE_DIR DEST_DIR)
  if (tensorflow_BUILD_LIBRARIES)
    string(LENGTH ${BASE_DIR} _base_len)
    math(EXPR _substr_start "${_base_len} + 1")
    foreach (_file ${${HDR_VAR}})
      if (${_file} MATCHES "\\.h$")
	get_filename_component(_dir ${_file} DIRECTORY)
	string(SUBSTRING ${_dir} ${_substr_start} -1 _outdir)
	install(FILES ${_file} DESTINATION ${DEST_DIR}/${_outdir})
      endif()
    endforeach()
  endif()
endfunction()

# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/tensorflow/contrib/cmake/external)

# Location where external projects will be downloaded
set (DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
     CACHE PATH "Location where external projects will be downloaded.")
mark_as_advanced(DOWNLOAD_LOCATION)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DEIGEN_AVOID_STL_ARRAY)
if(WIN32)
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC -D__VERSION__=\"MSVC\")
  add_definitions(-DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS)
  add_definitions(-DTENSORFLOW_USE_EIGEN_THREADPOOL -DEIGEN_HAS_C99_MATH -D_ITERATOR_DEBUG_LEVEL=0)
  add_definitions(-DNDEBUG /O2)  # Equivalent of -c opt in Bazel.
  add_definitions(/bigobj /nologo /EHsc /GF /FC /MP /Gm-)
  # Suppress warnings to reduce build log size.
  add_definitions(/wd4267 /wd4244 /wd4800 /wd4503 /wd4554 /wd4996 /wd4348 /wd4018)
  add_definitions(/wd4099 /wd4146 /wd4267 /wd4305 /wd4307)
  add_definitions(/wd4715 /wd4722 /wd4723 /wd4838 /wd4309 /wd4334)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -std=c++11")
endif()

if (tensorflow_OPTIMIZE_FOR_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
endif()

# External dependencies
if (tensorflow_BUILD_DEPENDENCIES)
  if(_tensorflow_need_zlib)
    include(zlib)
  endif()
  if(tensorflow_ENABLE_GIF)
    include(gif)
  endif()
  if(tensorflow_ENABLE_PNG)
    include(png)
  endif()
  if(tensorflow_ENABLE_JPEG)
    include(jpeg)
  endif()
  include(eigen)
  include(gemmlowp)
  include(jsoncpp)
  include(farmhash)
  include(highwayhash)
  include(protobuf)
  if (tensorflow_BUILD_CC_TESTS)
    include(googletest)
  endif()

  set(tensorflow_EXTERNAL_LIBRARIES
    ${zlib_STATIC_LIBRARIES}
    ${gif_STATIC_LIBRARIES}
    ${png_STATIC_LIBRARIES}
    ${jpeg_STATIC_LIBRARIES}
    ${jsoncpp_STATIC_LIBRARIES}
    ${farmhash_STATIC_LIBRARIES}
    ${highwayhash_STATIC_LIBRARIES}
    ${protobuf_STATIC_LIBRARIES}
  )
  set(tensorflow_EXTERNAL_DEPENDENCIES
    jsoncpp
    farmhash_copy_headers_to_destination
    highwayhash_copy_headers_to_destination
    protobuf
    eigen
    gemmlowp
    )
  if(_tensorflow_need_zlib)
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES zlib_copy_headers_to_destination)
  endif()
  if(tensorflow_ENABLE_GIF)
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES gif_copy_headers_to_destination)
  endif()
  if(tensorflow_ENABLE_PNG)
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES png_copy_headers_to_destination)
  endif()
  if(tensorflow_ENABLE_JPEG)
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES jpeg_copy_headers_to_destination)
  endif()

  include_directories(
    # Source and generated code.
    ${tensorflow_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${zlib_INCLUDE_DIR}
    ${gif_INCLUDE_DIR}
    ${png_INCLUDE_DIR}
    ${jpeg_INCLUDE_DIR}
    ${eigen_INCLUDE_DIRS}
    ${gemmlowp_INCLUDE_DIR}
    ${jsoncpp_INCLUDE_DIR}
    ${farmhash_INCLUDE_DIR}
    ${highwayhash_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
  )

  if(tensorflow_ENABLE_SSL_SUPPORT)
    include(boringssl)
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${boringssl_STATIC_LIBRARIES})
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES boringssl)
    include_directories(${boringssl_INCLUDE_DIR})
  endif()
  if(tensorflow_ENABLE_GRPC_SUPPORT)
    include(grpc)
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${grpc_STATIC_LIBRARIES})
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES grpc)
    include_directories(${GRPC_INCLUDE_DIRS})
  endif()
else()
  find_package(PkgConfig)
  if (_tensorflow_need_zlib)
    find_package(ZLIB REQUIRED)
  endif()
  if (tensorflow_ENABLE_PNG)
    pkg_check_modules(PNG libpng12)
  endif()
  if (tensorflow_ENABLE_GIF)
    pkg_check_modules(GIF gif)
  endif()
  if (tensorflow_ENABLE_JPEG)
    find_package(JPEG REQUIRED)
  endif()
  pkg_check_modules(JSONCPP jsoncpp)
  pkg_check_modules(EIGEN3 eigen33-60578b474802)
  find_library(GEMMLOWP_LIBRARIES eight_bit_int_gemm)
  find_path(_gemmlowp_include_dir gemmlowp/public/gemmlowp.h)
  set(GEMMLOWP_INCLUDE_DIR "${_gemmlowp_include_dir}/gemmlowp" CACHE PATH "gemmlow include dir")
  find_library(FARMHASH_LIBRARIES farmhash)
  find_library(HIGHWAYHASH_LIBRARIES highwayhash)
  find_path(HIGHWAYHASH_INCLUDE_DIR highwayhash/sip_hash.h)
  find_package(Protobuf REQUIRED)
  if (tensorflow_ENABLE_GRPC_SUPPORT)
    find_package(grpc REQUIRED)
  endif()
  if (tensorflow_ENABLE_SSL_SUPPORT)
    find_package(OpenSSL REQUIRED)
  endif()
  set(tensorflow_EXTERNAL_LIBRARIES
    ${ZLIB_LIBRARIES}
    ${PNG_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${GIF_LIBRARIES}
    ${JSONCPP_LDFLAGS}
    ${EIGEN3_LIBRARIES}
    ${GEMMLOWP_LIBRARIES}
    ${FARMHASH_LIBRARIES}
    ${HIGHWAYHASH_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    ${OPENSSL_LIBRARIES}
    ${GRPC_STATIC_LIBRARIES}
  )
  if (tensorflow_BUILD_CC_TESTS)
    find_package(googletest REQUIRED)
  endif()
  include_directories(
    # Source and generated code.
    ${tensorflow_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${tensorflow_SOURCE_DIR}/third_party/eigen3
    # External dependencies.
    ${ZLIB_INCLUDE_DIR}
    ${PNG_INCLUDE_DIRS}
    ${JPEG_INCLUDE_DIR}
    ${GIF_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
    ${GEMMLOWP_INCLUDE_DIR}
    ${JSONCPP_INCLUDE_DIR}
    ${HIGHWAYHASH_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
  )
endif()
  
if(WIN32)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES wsock32 ws2_32 shlwapi)
endif()
if(UNIX)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
endif()

if (tensorflow_ENABLE_GPU)
  if (WIN32)
    find_package(CUDA 8.0 REQUIRED)

    # by default we assume compute cabability 3.5 and 5.2. If you change this change it in
    # CUDA_NVCC_FLAGS and cuda_config.h below
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=\"sm_30,compute_30\";-gencode arch=compute_35,code=\"sm_35,compute_35\";-gencode arch=compute_52,code=\"sm_52,compute_52\")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--include-path ${PROJECT_BINARY_DIR}/$\{build_configuration\};--expt-relaxed-constexpr)
    set(CUDA_INCLUDE ${CUDA_TOOLKIT_TARGET_DIR} ${CUDA_TOOLKIT_TARGET_DIR}/extras/CUPTI/include)
    include_directories(${CUDA_INCLUDE})
    add_definitions(-DGOOGLE_CUDA=1 -DTF_EXTRA_CUDA_CAPABILITIES=3.0,3.5,5.2)

    # add cudnn
    include_directories(${CUDNN_HOME})
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDNN_HOME}/lib/x64/cudnn.lib)

    # create cuda_config.h
    FILE(WRITE ${tensorflow_SOURCE_DIR}/third_party/gpus/cuda/cuda_config.h
      "#ifndef CUDA_CUDA_CONFIG_H_\n"
      "#define CUDA_CUDA_CONFIG_H_\n"
      "#define TF_CUDA_CAPABILITIES CudaVersion(\"3.0\"),CudaVersion(\"3.5\"),CudaVersion(\"5.2\")\n"
      "#define TF_CUDA_VERSION \"64_80\"\n"
      "#define TF_CUDNN_VERSION \"64_5\"\n"
      "#define TF_CUDA_TOOLKIT_PATH \"${CUDA_TOOLKIT_ROOT_DIR}\"\n"
      "#endif  // CUDA_CUDA_CONFIG_H_\n"
    )

    # tf assumes in various places header files to be in cuda/include. On windows the cuda sdk
    # installs them under cuda/version/include and to avoid that we need to change tf we copy a
    # few files to cuda/include
    FILE(COPY
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda.h ${CUDA_TOOLKIT_TARGET_DIR}/include/cuComplex.h
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cublas_v2.h ${CUDNN_HOME}/include/cudnn.h
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cufft.h ${CUDA_TOOLKIT_TARGET_DIR}/include/curand.h
      DESTINATION ${tensorflow_SOURCE_DIR}/third_party/gpus/cuda/include
    )
    include_directories(${tensorflow_SOURCE_DIR}/third_party/gpus)
    # add cuda libraries to tensorflow_EXTERNAL_LIBRARIES
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CUDA_LIBRARIES})
  else()
    find_package(CUDA 8.0 REQUIRED)
    find_library(CUDNN_LIBRARY NAMES cudnn libcudnn cudnn_static_v6 libcudnn_static_v6 cudnn_static_v5 libcudnn_static_v5)
    # now identify which version
    find_path(CUDNN_HDRPATH cudnn_v6.h)
    if(EXISTS ${CUDNN_HDRPATH})
      set(CUDNN_HEADER "cudnn_v6.h")
      set(CUDNN_MAJVER 6)
    else()
      find_path(CUDNN_HDRPATH cudnn_v5.h)
      if(EXISTS ${CUDNN_HDRPATH})
        set(CUDNN_HEADER "cudnn_v5.h")
        set(CUDNN_MAJVER 5)
      endif()
    endif()

    # by default we assume compute cabability 5.3. If you change this change it in
    # cuda_config.h below
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--include-path ${PROJECT_BINARY_DIR}/$\{build_configuration\};--expt-relaxed-constexpr)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DEIGEN_HAS_VARIADIC_TEMPLATES=0;-Xcompiler -fPIC)
    include_directories(${CUDA_TOOLKIT_INCLUDE} ${CUDA_TOOLKIT_TARGET_DIR}/extras/CUPTI/include)
    add_definitions(-DGOOGLE_CUDA=1 -DTF_EXTRA_CUDA_CAPABILITIES=5.3)

    # create cuda_config.h
    FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/cuda_config.h
      "#ifndef CUDA_CUDA_CONFIG_H_\n"
      "#define CUDA_CUDA_CONFIG_H_\n"
      "#define TF_CUDA_CAPABILITIES CudaVersion(\"5.3\")\n"
      "#define TF_CUDA_VERSION \"8\"\n"
      "#define TF_CUDNN_VERSION \"${CUDNN_MAJVER}\"\n"
      "#define TF_CUDA_TOOLKIT_PATH \"${CUDA_TOOLKIT_ROOT_DIR}\"\n"
      "#endif  // CUDA_CUDA_CONFIG_H_\n"
    )

    # tf uses 'cudnn.h', but the actual header is versioned
    FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/include/cudnn.h
      "#include <${CUDNN_HEADER}>"
    )
    FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/include/cuda.h
      "#include <cuda.h>"
    )
    FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/include/cublas_v2.h
      "#include <cublas_v2.h>"
    )
    FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/include/cuComplex.h
      "#include <cuComplex.h>"
    )
    FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/include/cufft.h
      "#include <cufft.h>"
    )
    FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/include/curand.h
      "#include <curand.h>"
    )
    FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/extras/CUPTI/include/cupti.h
      "#include <cupti.h>"
    )
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus)
    # add cuda libraries to tensorflow_EXTERNAL_LIBRARIES
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})
  endif()
endif()

# Let's get to work!
include(tf_core_framework)
include(tf_tools)
if (tensorflow_ENABLE_GPU)
  include(tf_stream_executor)
endif()

include(tf_core_cpu)
include(tf_core_ops)
include(tf_core_direct_session)
include(tf_core_kernels)
if(tensorflow_ENABLE_GRPC_SUPPORT)
  include(tf_core_distributed_runtime)
endif()
include(tf_cc_ops)
if(tensorflow_BUILD_CC_EXAMPLE)
  include(tf_tutorials)
  include(tf_label_image_example)
endif()
if(tensorflow_BUILD_PYTHON_BINDINGS)
  include(tensorboard)
  include(tf_python)
endif()
if (tensorflow_BUILD_CC_TESTS OR tensorflow_BUILD_PYTHON_TESTS)
  include(tf_tests)
endif()

if (tensorflow_BUILD_LIBRARIES)
  if (tensorflow_BUILD_ALL_KERNELS)
    set(tf_kernel_objects $<TARGET_OBJECTS:tf_core_kernels>)
  else ()
    set(tf_kernel_objects
      $<TARGET_OBJECTS:tf_core_kernels_android>
      $<TARGET_OBJECTS:tf_core_kernels>)
  endif ()
  if (tensorflow_ENABLE_GPU)
    set(tf_stream_exec_objects $<TARGET_OBJECTS:tf_stream_executor>)
  else()
    set(tf_stream_exec_objects)
  endif()
  add_library(tensorflow SHARED
    $<TARGET_OBJECTS:tf_core_ops>
    $<TARGET_OBJECTS:tf_core_cpu>
    $<TARGET_OBJECTS:tf_core_framework>
    $<TARGET_OBJECTS:tf_core_lib>
    $<TARGET_OBJECTS:tf_cc_framework>
    $<TARGET_OBJECTS:tf_cc_ops>
    $<TARGET_OBJECTS:tf_core_direct_session>
    ${tf_kernel_objects}
    ${tf_stream_exec_objects}
    $<TARGET_OBJECTS:tf_protos_cc_objs>
    "${tensorflow_SOURCE_DIR}/tensorflow/c/c_api.cc"
  )
  set_target_properties(tensorflow PROPERTIES VERSION ${tensorflow_VERSION} SOVERSION ${tensorflow_VERSION_MAJOR})
  if (tensorflow_ENABLE_GPU)
    target_link_libraries(tensorflow PRIVATE ${tf_core_gpu_kernels_lib})
  endif()
  target_link_libraries(tensorflow PRIVATE ${tensorflow_EXTERNAL_LIBRARIES})
  install(TARGETS tensorflow LIBRARY DESTINATION lib)
  set(tensorflow_LIB_REFS "-ltensorflow")
  set(tensorflow_c_hdrs "${tensorflow_SOURCE_DIR}/tensorflow/c/c_api.h")
  InstallTFHeaders(tensorflow_c_hdrs ${tensorflow_SOURCE_DIR} include)

  install(DIRECTORY third_party/eigen3 DESTINATION include/tensorflow/third_party)

  configure_file(${tensorflow_SOURCE_DIR}/tensorflow.pc.in
                 ${CMAKE_CURRENT_BINARY_DIR}/tensorflow.pc @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tensorflow.pc DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
endif()

# Not strictly needed, but makes it easier to inspect the link command
string(REPLACE "<OBJECTS>" "-Wl,--whole-archive <OBJECT_DIR>/objects.a -Wl,--no-whole-archive" CMAKE_CXX_CREATE_SHARED_LIBRARY "${CMAKE_CXX_CREATE_SHARED_LIBRARY}")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY
  "<CMAKE_COMMAND> -E remove -f <OBJECT_DIR>/objects.a"
  "<CMAKE_AR> cr <OBJECT_DIR>/objects.a <OBJECTS>"
  "${CMAKE_CXX_CREATE_SHARED_LIBRARY}"
)

# For OE native builds, our LDFLAGS confuses CMake into stripping out
# rpath and -L flags that we actually need
if ("${CMAKE_SYSROOT}" STREQUAL "")
  if (NOT "${CMAKE_INSTALL_PREFIX}" STREQUAL "")
    list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib")
    list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  endif()
endif()
