cmake_minimum_required(VERSION 3.16)
project(kws_pipeline_consumer LANGUAGES C)

find_package(KwsPipeline CONFIG REQUIRED)

add_executable(kws_pipeline_consumer main.c)
target_link_libraries(kws_pipeline_consumer PRIVATE KwsPipeline::core)
target_compile_features(kws_pipeline_consumer PRIVATE c_std_11)
