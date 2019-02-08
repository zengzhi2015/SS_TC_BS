cmake_minimum_required(VERSION 3.13)
project(SWCD_SS)


set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR /home/opencv4/installation/OpenCV-master/lib/cmake/opencv4)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Threads REQUIRED)
if(THREADS_HAVE_PTHREAD_ARG)
    set_property(TARGET SWCD_SS PROPERTY COMPILE_OPTIONS "-pthread")
    set_property(TARGET SWCD_SS PROPERTY INTERFACE_COMPILE_OPTIONS "-pthread")
endif()


add_executable(SWCD_SS main.cpp BACKSUB.cpp Functions.cpp)

if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(SWCD_SS ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})
endif()