cmake_minimum_required(VERSION 3.18)
project(caesar_reader)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add the executable
add_executable(caesar_reader main.cpp)

# Link against LibTorch
target_link_libraries(caesar_reader "${TORCH_LIBRARIES}")

# Link filesystem library for C++17
target_link_libraries(caesar_reader stdc++fs)

set_property(TARGET caesar_reader PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET caesar_reader
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:caesar_reader>)
endif (MSVC)
