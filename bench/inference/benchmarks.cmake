# End-to-end product benchmark: public Engine API and native .ninfer artifacts only.
add_executable(ninfer_bench
  "${CMAKE_CURRENT_LIST_DIR}/ninfer_bench.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ninfer_bench_support.cpp")
ninfer_internal_includes(ninfer_bench)
target_include_directories(ninfer_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ninfer_bench PRIVATE NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(ninfer_bench PRIVATE ninfer_engine ${NINFER_CUDART_TARGET})

add_executable(ninfer_spec_router_calibration_bench
  "${CMAKE_CURRENT_LIST_DIR}/spec_router_calibration.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ninfer_bench_support.cpp")
ninfer_internal_includes(ninfer_spec_router_calibration_bench)
target_include_directories(ninfer_spec_router_calibration_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ninfer_spec_router_calibration_bench PRIVATE NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(ninfer_spec_router_calibration_bench PRIVATE ninfer_engine ninfer::json ${NINFER_CUDART_TARGET})

# The developer benchmark runs the same real-model correctness cases with its resident owner.
if(BUILD_TESTING)
  target_sources(ninfer_spec_router_calibration_bench PRIVATE
    "${PROJECT_SOURCE_DIR}/tests/models/qwen3_5/test_engine_calibrated_routing_real.cpp")
  target_include_directories(ninfer_spec_router_calibration_bench PRIVATE
    "${PROJECT_SOURCE_DIR}/tests")
  target_compile_definitions(ninfer_spec_router_calibration_bench PRIVATE
    NINFER_CALIBRATED_SUITE_LIBRARY=1)
  target_link_libraries(ninfer_spec_router_calibration_bench PRIVATE ninfer_runtime_support)
endif()
