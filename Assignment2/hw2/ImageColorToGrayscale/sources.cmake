add_lab("ImageColorToGrayscale")
add_lab_solution("ImageColorToGrayscale" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("ImageColorToGrayscale" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
