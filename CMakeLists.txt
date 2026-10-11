cmake_minimum_required(VERSION 3.20)
project(drowsiness_alert CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)

find_package(OpenCV REQUIRED)

add_library(drowsy STATIC src/drowsy.cpp)
target_include_directories(drowsy PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(drowsy PUBLIC ${OpenCV_LIBS})

enable_testing()
add_executable(test_drowsy tests/test_drowsy.cpp)
target_link_libraries(test_drowsy drowsy)
add_test(NAME test_drowsy COMMAND test_drowsy)

add_executable(drowsiness-alert src/main.cpp)
target_link_libraries(drowsiness-alert drowsy)
