# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(tensorflow C CXX)

set(tensorflow_VERSION_MAJOR "1")
set(tensorflow_VERSION_MINOR "4")
set(tensorflow_VERSION_PATCH "0")
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
set(CMAKE_CXX_STANDARD 14)

list(INSERT CMAKE_MODULE_PATH 0 "${tensorflow_SOURCE_DIR}/tensorflow/contrib/cmake")

# For compatibility with the original rules that had CMakeLists.txt
# in with the cmake modules, rather than at the top of the tree
set(tensorflow_source_dir "${tensorflow_SOURCE_DIR}")

# Options
option(tensorflow_VERBOSE "Enable for verbose output" OFF)
option(tensorflow_ENABLE_GPU "Enable GPU support" OFF)
option(tensorflow_ENABLE_SSL_SUPPORT "Enable boringssl support" OFF)
option(tensorflow_ENABLE_GRPC_SUPPORT "Enable gRPC support" ON)
option(tensorflow_ENABLE_HDFS_SUPPORT "Enable HDFS support" OFF)
option(tensorflow_ENABLE_JEMALLOC_SUPPORT "Enable jemalloc support" OFF)
option(tensorflow_ENABLE_AUDIO_SUPPORT "Enable audio support (requires fft2d)" OFF)
option(tensorflow_BUILD_CC_EXAMPLE "Build the C++ tutorial example" ON)
option(tensorflow_BUILD_PYTHON_BINDINGS "Build the Python bindings" ON)
option(tensorflow_BUILD_ALL_KERNELS "Build all OpKernels" ON)
option(tensorflow_BUILD_CONTRIB_KERNELS "Build OpKernels from tensorflow/contrib/..." ON)
option(tensorflow_BUILD_CC_TESTS "Build cc unit tests " OFF)
option(tensorflow_BUILD_PYTHON_TESTS "Build python unit tests " OFF)
option(tensorflow_BUILD_MORE_PYTHON_TESTS "Build more python unit tests for contrib packages" OFF)
option(tensorflow_BUILD_SHARED_LIB "Build TensorFlow as a shared library" OFF)
option(tensorflow_OPTIMIZE_FOR_NATIVE_ARCH "Enable compiler optimizations for the native processor architecture (if available)" ON)
option(tensorflow_WIN_CPU_SIMD_OPTIONS "Enables CPU SIMD instructions")
option(tensorflow_ENABLE_SNAPPY_SUPPORT "Enable SNAPPY compression support" ON)
option(tensorflow_BUILD_DEPENDENCIES "Download and build dependent packages" OFF)
option(tensorflow_BUILD_PROTO_TEXT "Build proto_text tool" ON)
option(tensorflow_ENABLE_GIF "Enable GIF support" OFF)
option(tensorflow_ENABLE_JPEG "Enable JPEG support" OFF)
option(tensorflow_ENABLE_PNG "Enable PNG support" OFF)
option(tensorflow_ENABLE_TYPES_SLIM "Reduce number of supported types" OFF)
option(tensorflow_ENABLE_SELECTIVE_REGISTRATION "Enable selective registration of ops" OFF)

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
  if (tensorflow_BUILD_SHARED_LIB)
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
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC)
  add_definitions(-DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS)
  add_definitions(-DTENSORFLOW_USE_EIGEN_THREADPOOL -DEIGEN_HAS_C99_MATH)
  add_definitions(-DTF_COMPILE_LIBRARY)
  add_definitions(/bigobj /nologo /EHsc /GF /MP /Gm-)
  # Suppress warnings to reduce build log size.
  add_definitions(/wd4267 /wd4244 /wd4800 /wd4503 /wd4554 /wd4996 /wd4348 /wd4018)
  add_definitions(/wd4099 /wd4146 /wd4267 /wd4305 /wd4307)
  add_definitions(/wd4715 /wd4722 /wd4723 /wd4838 /wd4309 /wd4334)
  add_definitions(/wd4003 /wd4244 /wd4267 /wd4503 /wd4506 /wd4800 /wd4996)
  # Suppress linker warnings.
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Ob2")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /D_ITERATOR_DEBUG_LEVEL=0")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -std=c++11")
endif()

if(tensorflow_ENABLE_AUDIO_SUPPORT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTF_AUDIO_SUPPORT")
endif()

if (tensorflow_OPTIMIZE_FOR_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
endif()

# MSVC SIMD instructions
if (tensorflow_WIN_CPU_SIMD_OPTIONS)
  if (WIN32)
    CHECK_CXX_COMPILER_FLAG("${tensorflow_WIN_CPU_SIMD_OPTIONS}" COMPILER_OPT_WIN_CPU_SIMD_SUPPORTED)
    if(COMPILER_OPT_WIN_CPU_SIMD_SUPPORTED)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${tensorflow_WIN_CPU_SIMD_OPTIONS}")
    else()
      message(FATAL_ERROR "${tensorflow_WIN_CPU_SIMD_OPTIONS} not supported")
    endif()
  endif()
endif()

if (tensorflow_ENABLE_JEMALLOC_SUPPORT)
  add_definitions(-DTENSORFLOW_USE_JEMALLOC -DJEMALLOC_EXPORT=)
endif()

# External dependencies
if (tensorflow_BUILD_DEPENDENCIES)
  include(zlib)
  include(gif)
  include(png)
  include(jpeg)
  include(lmdb)
  include(eigen)
  include(gemmlowp)
  include(jsoncpp)
  include(farmhash)
  include(fft2d)
  include(highwayhash)
  include(nsync)
  include(protobuf)
  include(re2)
  include(sqlite)
  if (tensorflow_BUILD_CC_TESTS)
    include(googletest)
  endif()

  set(_tf_PROTOBUF_DEPENDENCY protobuf)

  set(tensorflow_EXTERNAL_LIBRARIES
    ${zlib_STATIC_LIBRARIES}
    ${gif_STATIC_LIBRARIES}
    ${png_STATIC_LIBRARIES}
    ${jpeg_STATIC_LIBRARIES}
    ${lmdb_STATIC_LIBRARIES}
    ${jsoncpp_STATIC_LIBRARIES}
    ${farmhash_STATIC_LIBRARIES}
    ${fft2d_STATIC_LIBRARIES}
    ${highwayhash_STATIC_LIBRARIES}
    ${nsync_STATIC_LIBRARIES}
    ${protobuf_STATIC_LIBRARIES}
    ${re2_STATIC_LIBRARIES}
    ${sqlite_STATIC_LIBRARIES}
  )
  set(tensorflow_EXTERNAL_DEPENDENCIES
    zlib_copy_headers_to_destination
    gif_copy_headers_to_destination
    png_copy_headers_to_destination
    jpeg_copy_headers_to_destination
    lmdb_copy_headers_to_destination
    jsoncpp
    farmhash_copy_headers_to_destination
    highwayhash_copy_headers_to_destination
    nsync_copy_headers_to_destination
    protobuf
    eigen
    gemmlowp
    fft2d
    re2
    sqlite_copy_headers_to_destination
  )

  include_directories(
    # Source and generated code.
    ${tensorflow_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${zlib_INCLUDE_DIR}
    ${gif_INCLUDE_DIR}
    ${png_INCLUDE_DIR}
    ${jpeg_INCLUDE_DIR}
    ${lmdb_INCLUDE_DIR}
    ${eigen_INCLUDE_DIRS}
    ${gemmlowp_INCLUDE_DIR}
    ${jsoncpp_INCLUDE_DIR}
    ${farmhash_INCLUDE_DIR}
    ${highwayhash_INCLUDE_DIR}
    ${nsync_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
    ${re2_INCLUDE_DIR}
    ${sqlite_INCLUDE_DIR}
  )

  if(tensorflow_ENABLE_SSL_SUPPORT)
    include(boringssl)
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${boringssl_STATIC_LIBRARIES})
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES boringssl)
    include_directories(${boringssl_INCLUDE_DIR})
  endif()
  if(tensorflow_ENABLE_GRPC_SUPPORT)
    include(grpc)
    set(_tf_GRPC_DEPENDENCY grpc)
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${grpc_STATIC_LIBRARIES})
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES grpc)
    include_directories(${GRPC_INCLUDE_DIRS})
  endif()
  if(tensorflow_ENABLE_JEMALLOC_SUPPORT)
    include(jemalloc)
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${jemalloc_STATIC_LIBRARIES})
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES jemalloc)
    include_directories(${jemalloc_INCLUDE_DIRS})
  endif()
  if(tensorflow_ENABLE_SNAPPY_SUPPORT)
    include(snappy)
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${snappy_STATIC_LIBRARIES})
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES snappy)
    include_directories(${snappy_INCLUDE_DIR})
  endif()
else()
  find_package(PkgConfig)
  find_package(ZLIB REQUIRED)
  if (tensorflow_ENABLE_PNG)
    pkg_check_modules(PNG libpng12)
  endif()
  if (tensorflow_ENABLE_GIF)
    find_library(GIF_LIBRARIES gif)
  endif()
  if (tensorflow_ENABLE_JPEG)
    find_package(JPEG REQUIRED)
  endif()
  find_library(LMDB_LIBRARIES lmdb)
  pkg_check_modules(JSONCPP jsoncpp)
  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/include/json/json.h)
    FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/include/json/json.h
      "#include <json/json.h>"
    )
  endif()

  pkg_check_modules(EIGEN3 eigen33-429aa5254200)
  get_filename_component(EIGEN3_INCLUDE_SUBDIR ${EIGEN3_INCLUDE_DIRS} NAME)
  find_library(GEMMLOWP_LIBRARIES eight_bit_int_gemm)
  find_path(_gemmlowp_include_dir gemmlowp/public/gemmlowp.h)
  set(GEMMLOWP_INCLUDE_DIR "${_gemmlowp_include_dir}/gemmlowp" CACHE PATH "gemmlow include dir")
  find_library(FARMHASH_LIBRARIES farmhash)
  if (tensorflow_ENABLE_AUDIO_SUPPORT)
    find_library(FFT2D_LIBRARIES fft2d)
  endif()
  find_library(HIGHWAYHASH_LIBRARIES highwayhash)
  find_path(HIGHWAYHASH_INCLUDE_DIR highwayhash/sip_hash.h)
  find_library(NSYNC_LIBRARIES nsync)
  find_package(Protobuf REQUIRED)
  pkg_check_modules(RE2 re2)
  pkg_check_modules(SQLITE sqlite3)
  if (tensorflow_ENABLE_GRPC_SUPPORT)
    include(gRPCConfig)
    set(GRPC_LDFLAGS -lgrpc++_unsecure -lgrpc_unsecure -lgpr -lcares)
  endif()
  if (tensorflow_ENABLE_SSL_SUPPORT)
    find_package(OpenSSL REQUIRED)
  endif()
  # XXX maybe jemalloc, snappy
  set(tensorflow_EXTERNAL_LIBRARIES
    ${ZLIB_LIBRARIES}
    ${PNG_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${GIF_LIBRARIES}
    ${LMDB_LIBRARIES}
    ${JSONCPP_LDFLAGS}
    ${EIGEN3_LIBRARIES}
    ${GEMMLOWP_LIBRARIES}
    ${FARMHASH_LIBRARIES}
    ${FFT2D_LIBRARIES}
    ${HIGHWAYHASH_LIBRARIES}
    ${NSYNC_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    ${RE2_LIBRARIES}
    ${SQLITE_LIBRARIES}
    ${OPENSSL_LIBRARIES}
    ${GRPC_LDFLAGS}
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
    ${JSONCPP_INCLUDE_DIRS}
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
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)  # Flush denormals to zero
    set(CUDA_INCLUDE ${CUDA_TOOLKIT_TARGET_DIR} ${CUDA_TOOLKIT_TARGET_DIR}/extras/CUPTI/include)
    include_directories(${CUDA_INCLUDE})
    add_definitions(-DGOOGLE_CUDA=1 -DTF_EXTRA_CUDA_CAPABILITIES=3.0,3.5,5.2)

    # add cudnn
    if(NOT CUDNN_HOME)
      set(CUDNN_HOME ${CUDA_TOOLKIT_TARGET_DIR})
    endif(NOT CUDNN_HOME)
    include_directories(${CUDNN_HOME})
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_curand_LIBRARY} ${CUDA_cupti_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDNN_HOME}/lib/x64/cudnn.lib)

    # create cuda_config.h
    FILE(WRITE ${tensorflow_SOURCE_DIR}/third_party/gpus/cuda/cuda_config.h
      "#ifndef CUDA_CUDA_CONFIG_H_\n"
      "#define CUDA_CUDA_CONFIG_H_\n"
      "#define TF_CUDA_CAPABILITIES CudaVersion(\"3.0\"),CudaVersion(\"3.5\"),CudaVersion(\"5.2\")\n"
      "#define TF_CUDA_VERSION \"64_80\"\n"
      "#define TF_CUDNN_VERSION \"64_6\"\n"
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
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cuda_runtime_api.h
      ${CUDA_TOOLKIT_TARGET_DIR}/include/cusolverDn.h
      DESTINATION ${tensorflow_source_dir}/third_party/gpus/cuda/include
    )
    include_directories(${tensorflow_SOURCE_DIR}/third_party/gpus)
    # add cuda libraries to tensorflow_EXTERNAL_LIBRARIES
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CUDA_LIBRARIES})

    # NOTE(mrry): Update these flags when the version of CUDA or cuDNN used
    # in the default build is upgraded.
    set(tensorflow_BUILD_INFO_FLAGS --build_config cuda --key_value
      msvcp_dll_name=msvcp140.dll
      cudart_dll_name=cudart64_80.dll
      cuda_version_number=8.0
      nvcuda_dll_name=nvcuda.dll
      cudnn_dll_name=cudnn64_6.dll
      cudnn_version_number=6)
  else(WIN32)
    find_package(CUDA 8.0 REQUIRED)
    find_library(CUDNN_LIBRARY NAMES cudnn libcudnn cudnn_static_v6 libcudnn_static_v6 cudnn_static_v5 libcudnn_static_v5)
    # now identify which version
    find_path(CUDNN_V6_HDRPATH cudnn_v6.h)
    if(EXISTS ${CUDNN_V6_HDRPATH})
      set(CUDNN_HEADER "cudnn_v6.h")
      set(CUDNN_MAJVER 6)
    else()
      find_path(CUDNN_V5_HDRPATH cudnn_v5.h)
      if(EXISTS ${CUDNN_V5_HDRPATH})
        set(CUDNN_HEADER "cudnn_v5.h")
        set(CUDNN_MAJVER 5)
      endif()
    endif()

    # by default we assume compute cabability 5.3. If you change this change it in
    # cuda_config.h below
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--include-path ${PROJECT_BINARY_DIR}/$\{build_configuration\};--expt-relaxed-constexpr)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)  # Flush denormals to zero
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DEIGEN_HAS_VARIADIC_TEMPLATES=0;-Xcompiler -fPIC)
    set(CUDA_INCLUDE ${CUDA_TOOLKIT_TARGET_DIR} ${CUDA_TOOLKIT_TARGET_DIR}/extras/CUPTI/include)
    include_directories(${CUDA_INCLUDE})
    add_definitions(-DGOOGLE_CUDA=1 -DTF_EXTRA_CUDA_CAPABILITIES=5.3)

    # create cuda_config.h
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/cuda_config.h)
      FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/cuda_config.h
        "#ifndef CUDA_CUDA_CONFIG_H_\n"
        "#define CUDA_CUDA_CONFIG_H_\n"
        "#define TF_CUDA_CAPABILITIES CudaVersion(\"5.3\")\n"
        "#define TF_CUDA_VERSION \"8.0\"\n"
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
      FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/include/cuda_runtime_api.h
        "#include <cuda_runtime_api.h>"
      )
      FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus/cuda/include/cusolverDn.h
        "#include <cusolverDn.h>"
      )
    endif()
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/gpus)
    # add cuda libraries to tensorflow_EXTERNAL_LIBRARIES
    list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})
  endif(WIN32)
  if (tensorflow_BUILD_DEPENDENCIES)
    include(cub)
    list(APPEND tensorflow_EXTERNAL_DEPENDENCIES cub)
  else()
    find_path(cub_INCLUDE_DIR cub/cub.cuh)
  endif()
    include_directories(${cub_INCLUDE_DIR})
else(tensorflow_ENABLE_GPU)
  if(WIN32)
    set(tensorflow_BUILD_INFO_FLAGS --build_config cpu --key_value
      msvcp_dll_name=msvcp140.dll)
  endif(WIN32)
endif(tensorflow_ENABLE_GPU)

# Find python executable
include(FindPythonInterp)
if(NOT ${PYTHONINTERP_FOUND})
    message(FATAL_ERROR "CMake was unable to find a python interpreter.")
endif()

# Let's get to work!
include(tf_core_framework)
# NOTE: Disabled until issue #3996 is fixed.
# include(tf_stream_executor)
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
# We include tf_cc_ops first, because tf_c depends on tf_cc.
include(tf_cc_ops)
include(tf_c)
include(tf_grappler)
include(tf_core_profiler)
if(tensorflow_BUILD_CC_EXAMPLE)
  include(tf_tutorials)
  include(tf_label_image_example)
endif()
include(tf_tools)
if(tensorflow_BUILD_PYTHON_BINDINGS)
  include(tf_python)
endif()
if(tensorflow_BUILD_SHARED_LIB)
  include(tf_shared_lib)
endif()
if(tensorflow_BUILD_CC_TESTS OR tensorflow_BUILD_PYTHON_TESTS)
  include(tf_tests)
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
