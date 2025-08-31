
add_executable(test_caesar
    testCaesarReader.cpp
)

target_link_libraries(test_caesar
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)


add_executable(test_bsconvu
    testBsconvu.cpp
)

target_link_libraries(test_bsconvu
    PRIVATE
        caesar_lib
        "${TORCH_LIBRARIES}"
        stdc++fs
)


if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET test_caesar
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:test_caesar>)
  add_custom_command(TARGET test_bsconvu
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:test_bsconvu>)
endif()

