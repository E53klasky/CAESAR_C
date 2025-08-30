# Build the main caesar library
add_library(caesar_lib
    compressor.cpp
    normalize.cpp
)

# Include current dir + pretrainedModelReader
target_include_directories(caesar_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/pretrainedModelIO
)

# Link Torch
target_link_libraries(caesar_lib
    PUBLIC
        "${TORCH_LIBRARIES}"
)

# Add the subdir for pretrained model reader (in case later we separate it)
add_subdirectory(pretrainedModelIO)

