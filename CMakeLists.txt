cmake_minimum_required(VERSION 3.15...3.27)
project(zombie)

option(ZOMBIE_BUILD_DEMO "Build demo" ON)
option(ZOMBIE_BUILD_BINDINGS "Build bindings" OFF)

################################################################################
if(ZOMBIE_BUILD_BINDINGS)
    if (CMAKE_VERSION VERSION_LESS 3.18)
        set(DEV_MODULE Development)
    else()
        set(DEV_MODULE Development.Module)
    endif()
    find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter ${DEV_MODULE}
        OPTIONAL_COMPONENTS Development.SABIModule)
endif()

# submodule check
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_BUILD_TYPE MATCHES RELEASE)
    add_definitions(-DNDEBUG)
endif()

################################################################################
# add dependencies
get_directory_property(hasParent PARENT_DIRECTORY)

set(TBB_TEST          OFF CACHE BOOL " " FORCE)
set(TBB_INSTALL       OFF CACHE BOOL " " FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb)

if(hasParent)
    set(ZOMBIE_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include PARENT_SCOPE)
    set(ZOMBIE_NANOFLANN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/nanoflann/include PARENT_SCOPE)
else()
    set(ZOMBIE_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include)
    set(ZOMBIE_NANOFLANN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/nanoflann/include)
endif()

set(FCPW_USE_ENOKI OFF CACHE BOOL "Enable CPU vectorization" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/fcpw)

list(APPEND ZOMBIE_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/pcg32)
list(APPEND ZOMBIE_DEPS_INCLUDES ${ZOMBIE_TBB_INCLUDES})
list(APPEND ZOMBIE_DEPS_INCLUDES ${ZOMBIE_NANOFLANN_INCLUDES})
list(APPEND ZOMBIE_DEPS_INCLUDES ${FCPW_EIGEN_INCLUDES})
list(APPEND ZOMBIE_DEPS_INCLUDES ${FCPW_ENOKI_INCLUDES})

################################################################################
# build library
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include> ${ZOMBIE_DEPS_INCLUDES})
target_link_libraries(${PROJECT_NAME} INTERFACE fcpw TBB::tbb)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

################################################################################
# build demo
if(ZOMBIE_BUILD_DEMO)
    add_subdirectory(demo)
endif()

# build Python bindings
if(ZOMBIE_BUILD_BINDINGS)
    add_subdirectory(deps/nanobind)
    add_subdirectory(python)
endif()

# SYCL configuration for NVIDIA GPU backend
set(SYCL_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda)

add_executable(hello_sycl sycl/hello_sycl.cpp)
target_compile_options(hello_sycl PRIVATE ${SYCL_FLAGS})
target_link_options(hello_sycl PRIVATE ${SYCL_FLAGS})

add_executable(zombie_sycl_smoke sycl/zombie_sycl_smoke.cpp)
target_link_libraries(zombie_sycl_smoke PRIVATE zombie)
target_compile_options(zombie_sycl_smoke PRIVATE ${SYCL_FLAGS})
target_link_options(zombie_sycl_smoke PRIVATE ${SYCL_FLAGS})

add_executable(walkstate_lite_sycl sycl/walkstate_lite_sycl.cpp)
target_compile_options(walkstate_lite_sycl PRIVATE ${SYCL_FLAGS})
target_link_options(walkstate_lite_sycl PRIVATE ${SYCL_FLAGS})

add_executable(walkstate_bridge_sycl sycl/walkstate_bridge_sycl.cpp)
target_link_libraries(walkstate_bridge_sycl PRIVATE zombie)
target_compile_options(walkstate_bridge_sycl PRIVATE ${SYCL_FLAGS})
target_link_options(walkstate_bridge_sycl PRIVATE ${SYCL_FLAGS})

add_executable(walktask_lite_sycl sycl/walktask_lite_sycl.cpp)
target_link_libraries(walktask_lite_sycl PRIVATE zombie)
target_compile_options(walktask_lite_sycl PRIVATE ${SYCL_FLAGS})
target_link_options(walktask_lite_sycl PRIVATE ${SYCL_FLAGS})

add_executable(walktask_phased_sycl sycl/walktask_phased_sycl.cpp)
target_link_libraries(walktask_phased_sycl PRIVATE zombie)
target_compile_options(walktask_phased_sycl PRIVATE ${SYCL_FLAGS})
target_link_options(walktask_phased_sycl PRIVATE ${SYCL_FLAGS})

add_executable(wos_analytic_test sycl/wos_analytic_test.cpp)
target_link_libraries(wos_analytic_test PRIVATE zombie)
target_compile_options(wos_analytic_test PRIVATE ${SYCL_FLAGS})
target_link_options(wos_analytic_test PRIVATE ${SYCL_FLAGS})

add_executable(wos_multipoint_test sycl/wos_multipoint_test.cpp)
target_link_libraries(wos_multipoint_test PRIVATE zombie)
target_compile_options(wos_multipoint_test PRIVATE ${SYCL_FLAGS})
target_link_options(wos_multipoint_test PRIVATE ${SYCL_FLAGS})

add_executable(wos_sycl_baseline sycl/wos_sycl_baseline.cpp)
target_link_libraries(wos_sycl_baseline PRIVATE zombie)
target_compile_options(wos_sycl_baseline PRIVATE ${SYCL_FLAGS})
target_link_options(wos_sycl_baseline PRIVATE ${SYCL_FLAGS})

add_executable(wos_sycl_dg sycl/wos_sycl_dg.cpp)
target_link_libraries(wos_sycl_dg PRIVATE zombie)
target_compile_options(wos_sycl_dg PRIVATE ${SYCL_FLAGS})
target_link_options(wos_sycl_dg PRIVATE ${SYCL_FLAGS})

add_executable(wos_sycl_mp sycl/wos_sycl_mp.cpp)
target_link_libraries(wos_sycl_mp PRIVATE zombie)
target_compile_options(wos_sycl_mp PRIVATE ${SYCL_FLAGS})
target_link_options(wos_sycl_mp PRIVATE ${SYCL_FLAGS})

add_executable(experiment_runner sycl/experiment_runner.cpp)
target_link_libraries(experiment_runner PRIVATE zombie)
target_compile_options(experiment_runner PRIVATE ${SYCL_FLAGS})
target_link_options(experiment_runner PRIVATE ${SYCL_FLAGS})