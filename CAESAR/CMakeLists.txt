find_package(MPI)
find_package(CUDA)
find_package(PkgConfig)
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

# JL Modified
# 1. Fontier: ZSTD / AOCL-Compression Detection Logic
set(AOCL_ROOT $ENV{OLCF_AOCL_COMPRESSION_ROOT}) # correct aocl -> will use aocl zstd
##set(AOCL_ROOT $ENV{AOCL_COMPRESSION_ROOT}) # dummy aocl -> will use system zstd
if(AOCL_ROOT)
    message(STATUS ">>> Found AOCL-Compression Environment: ${AOCL_ROOT}")
    set(ZSTD_INCLUDE_DIR "${AOCL_ROOT}/include")

    find_library(ZSTD_LIBRARY 
        NAMES aocl_compression zstd 
        HINTS "${AOCL_ROOT}/lib"
        NO_DEFAULT_PATH
    )
    
    if(ZSTD_LIBRARY)
        set(ZSTD_FOUND TRUE)
        message(STATUS ">>> Using AOCL Optimized Zstd: ${ZSTD_LIBRARY}")
    endif()
endif()

# If AOCL not found, system default ZSTD search
if(NOT ZSTD_FOUND)
    if(PkgConfig_FOUND)
        pkg_check_modules(ZSTD IMPORTED_TARGET libzstd)
    endif()

    if(NOT ZSTD_FOUND)
        find_path(ZSTD_INCLUDE_DIR zstd.h)
        find_library(ZSTD_LIBRARY NAMES zstd)
        if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
            set(ZSTD_FOUND TRUE)
            message(STATUS "Found Standard Zstd: ${ZSTD_LIBRARY}")
        else()
            message(FATAL_ERROR "Zstd library not found. Please load 'aocl-compression' module or install libzstd-dev")
        endif()
    endif()
endif()

# 2. GPU (CUDA/HIP) & NVCOMP Configuration
# GPU operation is enabled if torch supports GPU
# This is to make sure that on Frontier, PCA is performed on GPU, whereas ZSTD is on CPU (AOCL)

if(TORCH_CUDA_ARCH_LIST OR CUDA_FOUND OR HIP_FOUND)
    set(USE_CUDA ON)
    message(STATUS "GPU Support Enabled (Torch/CUDA/HIP detected)")
    
    if(HIP_FOUND)
        message(STATUS ">>> ROCm (HIP) Environment Detected")
    endif()
else()
    set(USE_CUDA OFF)
    message(STATUS "GPU Support Disabled (Using CPU Only)")
endif()

set(NVCOMP_FOUND OFF)

# If GPU enabled, then search for nvcomp
if(USE_CUDA)
    message(STATUS "Checking for nvcomp (NVIDIA specific)...")

    find_path(NVCOMP_INCLUDE_DIR
        NAMES nvcomp.h nvcomp/nvcomp.h
        HINTS ENV NVCOMP_ROOT ENV NVCOMP_DIR
        PATH_SUFFIXES include
    )

    find_library(NVCOMP_LIBRARY
        NAMES nvcomp
        HINTS ENV NVCOMP_ROOT ENV NVCOMP_DIR
        PATH_SUFFIXES lib lib64
    )

    if(NVCOMP_INCLUDE_DIR AND NVCOMP_LIBRARY)
        set(NVCOMP_FOUND TRUE)
        message(STATUS ">>> Found nvcomp: ${NVCOMP_LIBRARY}")
        message(STATUS ">>> nvcomp include dir: ${NVCOMP_INCLUDE_DIR}")
    else()
        message(STATUS ">>> nvcomp NOT found. Will use Hybrid Mode (GPU PCA + CPU ZSTD/AOCL).")
    endif()
endif()

# ------------------------------------------------------------------------------
set(CAESAR_SRC
    CAESAR.cpp
    dataset/dataset.cpp
    models/range_coder/rans_coder.cpp
    models/caesar_compress.cpp
    models/caesar_decompress.cpp
    models/model_utils.cpp
    models/runGaeCuda.cpp  
    data_utils.cpp
)

add_library(caesar_lib SHARED ${CAESAR_SRC})

target_compile_definitions(caesar_lib PRIVATE
    DEFAULT_CAESAR_MODEL_DIR="${CMAKE_INSTALL_PREFIX}/share/caesar/models"
)

# [USE_CUDA Configuration]
if(USE_CUDA)
    target_compile_definitions(caesar_lib PUBLIC USE_CUDA)
    message(STATUS "Build defined with USE_CUDA")

    if(HIP_FOUND)
        target_compile_definitions(caesar_lib PUBLIC USE_ROCM)
        
        if(DEFINED ENV{ROCM_PATH})
            target_include_directories(caesar_lib PRIVATE $ENV{ROCM_PATH}/include)
        elseif(EXISTS "/opt/rocm/include")
            target_include_directories(caesar_lib PRIVATE "/opt/rocm/include")
        endif()
    endif()
endif()

# [NVCOMP Configuration]
if(NVCOMP_FOUND)
    target_compile_definitions(caesar_lib PRIVATE ENABLE_NVCOMP)
    message(STATUS "Build defined with ENABLE_NVCOMP")
endif()

target_include_directories(caesar_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dataset>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/range_coder>
        $<INSTALL_INTERFACE:include/caesar>
        ${ZSTD_INCLUDE_DIR}
)

target_link_libraries(caesar_lib PUBLIC torch ${ZSTD_LIBRARY})

# GPU Libraries Linking
if(USE_CUDA)
    if(CUDA_FOUND)
        target_include_directories(caesar_lib PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(caesar_lib PUBLIC ${CUDA_LIBRARIES})
    endif()
    
    if(NVCOMP_FOUND)
        target_include_directories(caesar_lib PRIVATE ${NVCOMP_INCLUDE_DIR})
        target_link_libraries(caesar_lib PUBLIC ${NVCOMP_LIBRARY})
    endif()
endif()

# ZSTD Linking Log
if(ZSTD_FOUND)
    message(STATUS "Final Link: Zstd Support Enabled")
endif()

if(MPI_FOUND)
    target_include_directories(caesar_lib PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(caesar_lib PUBLIC MPI::MPI_CXX)
    target_compile_definitions(caesar_lib PUBLIC USE_MPI)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(caesar_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(caesar_lib PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -mtune=native>
)

if(USE_CUDA)
    target_compile_options(caesar_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3 --maxrregcount=64>
    )
endif()

set_target_properties(caesar_lib PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)