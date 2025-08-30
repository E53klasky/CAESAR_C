# Add subdirectory for BCRN
add_subdirectory(BCRN)

# Add compressModules3dMidSR library
add_library(compressModules3dMidSR
    compressModules3dMidSR.cpp
    compressModules3dMidSR.h
)

target_include_directories(compressModules3dMidSR
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Link Torch
target_link_libraries(compressModules3dMidSR
    PUBLIC "${TORCH_LIBRARIES}"
)

# Optionally link to caesar_lib if needed
target_link_libraries(compressModules3dMidSR
    PUBLIC caesar_lib
)

