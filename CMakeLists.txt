cmake_minimum_required(VERSION 3.17)

project(AOC2020_problem1 CUDA)

set(CMAKE_CUDA_STANDARD 14)
add_executable(AOC2020_problem1 main.cu)


set_target_properties(
        AOC2020_problem1
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES OFF)