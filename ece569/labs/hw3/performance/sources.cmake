add_lab_solution("BasicPerformance" ${CMAKE_CURRENT_LIST_DIR}/basic_solution.cu)
add_lab_solution("TiledPerformance" ${CMAKE_CURRENT_LIST_DIR}/tiled_solution.cu)
add_generator("BasicPerformance" ${CMAKE_CURRENT_LIST_DIR}/basic_generator.cpp)
add_generator("TiledPerformance" ${CMAKE_CURRENT_LIST_DIR}/tiled_generator.cpp)
