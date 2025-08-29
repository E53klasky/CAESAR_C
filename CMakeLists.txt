cmake_minimum_required(VERSION 3.18)
project(caesar_reader)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Torch REQUIRED)


add_subdirectory(CAESAR)


add_executable(caesar_reader main.cpp)


target_link_libraries(caesar_reader
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)


if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET caesar_reader
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:caesar_reader>)
endif ()

