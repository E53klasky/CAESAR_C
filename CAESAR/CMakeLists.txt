
add_library(caesar_lib
    compressor.cpp
)


target_include_directories(caesar_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)


target_link_libraries(caesar_lib
    PUBLIC
        "${TORCH_LIBRARIES}"
)

