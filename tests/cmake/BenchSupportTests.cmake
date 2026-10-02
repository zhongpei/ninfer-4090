ninfer_add_test(ninfer_bench_support_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_ninfer_bench_support.cpp"
          ${PROJECT_SOURCE_DIR}/bench/inference/ninfer_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer::json)

target_include_directories(ninfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/inference)

add_executable(ninfer_context_cost_measure_test
  "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost_measure.cpp"
  ${PROJECT_SOURCE_DIR}/bench/context_cost/context_cost_measure.cpp)

target_include_directories(ninfer_context_cost_measure_test PRIVATE
  ${PROJECT_SOURCE_DIR}/bench/context_cost)

add_test(NAME ninfer_context_cost_measure_test COMMAND ninfer_context_cost_measure_test)

# CPU-only response/integrity regressions and an owned loopback HTTP process fixture; no GPU.
add_test(NAME ninfer_server_ab_response_test
  COMMAND "${Python3_EXECUTABLE}" -m unittest discover
          -s tests/tools -p test_server_ab_response.py -v)
set_tests_properties(ninfer_server_ab_response_test PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  TIMEOUT 60)
