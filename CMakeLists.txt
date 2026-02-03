cmake_minimum_required(VERSION 3.5)

include_directories(${AADC_SOURCE_DIR}/include)
include_directories(${AADC_SOURCE_DIR}/3rdparty)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)
# include_directories(${AADC_SOURCE_DIR}/3rdparty/adept)  # Adept removed

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")

find_package (Threads)

add_executable(xva_server XVAJobRequest.cpp XVAServer.cpp)

find_library(AADC_LIB NAMES aadc-avx2 aadc PATHS ${AADC_SOURCE_DIR}/lib)
target_link_libraries(xva_server ${AADC_LIB})

target_link_libraries (xva_server ${CMAKE_THREAD_LIBS_INIT})
