# Build BCRN library
add_library(BCRN
   BSConvU.h
  
    blocks.cpp
    blocks.h
    BSConvU.h
)

# Include current dir headers
target_include_directories(BCRN
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Link Torch
target_link_libraries(BCRN
    PUBLIC "${TORCH_LIBRARIES}"
)

# Optionally link to caesar_lib or compressModules3dMidSR if BCRN depends on them
target_link_libraries(BCRN
    PUBLIC caesar_lib
    PUBLIC compressModules3dMidSR
)

