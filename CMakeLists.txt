cmake_minimum_required(VERSION 3.0)
project(MLPP)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)

# Include header files
include_directories(include)

# Add source files
file(GLOB SOURCES "src/*.cpp")

# Define the executable
add_executable(MLPP.exe ${SOURCES})

# Debug configuration
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Werror -O0")

# Release configuration
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add subdirectories for external libraries if needed
# add_subdirectory(libs/someLibrary)

# Link libraries

