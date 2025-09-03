# Build BCRN library
add_library(BCRN
    blocks.cpp
   bcrnModel.cpp
)

# Include current dir headers
target_include_directories(BCRN
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Link Torch
target_link_libraries(BCRN
    PUBLIC "${TORCH_LIBRARIES}"
)

# Link dependencies
target_link_libraries(BCRN
    PUBLIC caesar_lib
)
# Note: Removed compressModules3dMidSR dependency to avoid circular deps
