cmake_minimum_required(VERSION 3.14)

project(face_test LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${ONNXRUNTIME_DIR}/include)
include_directories(${OPENCV_DIR}/include/opencv4)

file(GLOB SRC_HEADERS
    "*.h"
    )

file(GLOB SRC_SOURCES
    "*.cpp"
    )

add_executable(face_test
  ${SRC_SOURCES}
  ${SRC_HEADERS}
)

# repair .so for target os
target_link_libraries(${PROJECT_NAME}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
    ${OPENCV_DIR}/lib/libopencv-sl_core.so
    ${OPENCV_DIR}/lib/libopencv-sl_dnn.so
    ${OPENCV_DIR}/lib/libopencv-sl_imgproc.so
    ${OPENCV_DIR}/lib/libopencv-sl_imgcodecs.so
    ${OPENCV_DIR}/lib/libopencv-sl_highgui.so)

install(TARGETS face_test
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
