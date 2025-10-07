# Set minimum CMake version for Metal support
cmake_minimum_required(VERSION 3.23)

# Detect Metal availability (macOS only)
if(NOT APPLE)
    message(FATAL_ERROR "Metal backend requires macOS")
endif()

project(PIE_METAL_KERNELS)

set(CMAKE_BUILD_TYPE Release)

# Define output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find Metal compiler
find_program(METAL_COMPILER xcrun REQUIRED)

# Find all Metal source files in the kernels directory
file(GLOB METAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.metal")

if(NOT METAL_SOURCES)
    message(FATAL_ERROR "No Metal source files found in ${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
endif()

message(STATUS "Found ${CMAKE_MATCH_0} Metal kernel files:")
foreach(METAL_SOURCE ${METAL_SOURCES})
    message(STATUS "  - ${METAL_SOURCE}")
endforeach()

# Create Metal library compilation target
set(METAL_AIR_FILES "")
foreach(METAL_SOURCE ${METAL_SOURCES})
    get_filename_component(KERNEL_NAME ${METAL_SOURCE} NAME_WE)
    set(AIR_FILE "${CMAKE_BINARY_DIR}/${KERNEL_NAME}.air")

    # Compile .metal to .air
    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND ${METAL_COMPILER} -sdk macosx metal -c ${METAL_SOURCE} -o ${AIR_FILE}
        DEPENDS ${METAL_SOURCE}
        COMMENT "Compiling Metal kernel: ${KERNEL_NAME}"
    )

    list(APPEND METAL_AIR_FILES ${AIR_FILE})
endforeach()

# Link all .air files into single .metallib
set(METALLIB_OUTPUT "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pie_metal_kernels.metallib")
add_custom_command(
    OUTPUT ${METALLIB_OUTPUT}
    COMMAND ${METAL_COMPILER} -sdk macosx metallib ${METAL_AIR_FILES} -o ${METALLIB_OUTPUT}
    DEPENDS ${METAL_AIR_FILES}
    COMMENT "Creating Metal library: pie_metal_kernels.metallib"
)

# Create custom target for Metal library (this is the main target)
add_custom_target(pie_metal_kernels ALL DEPENDS ${METALLIB_OUTPUT})

# Print success message
add_custom_command(TARGET pie_metal_kernels POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "✅ Metal kernels compiled successfully:"
    COMMAND ${CMAKE_COMMAND} -E echo "   Output: ${METALLIB_OUTPUT}"
    COMMENT "Metal compilation completed"
)

# Test target to verify metallib was created
add_custom_target(test_metallib
    COMMAND test -f "${METALLIB_OUTPUT}" && echo "✅ metallib file exists" || echo "❌ metallib file not found"
    DEPENDS pie_metal_kernels
    COMMENT "Testing metallib creation"
)