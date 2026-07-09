add_ops_compile_options(
    OP_NAME KvCacheBlockGather
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
    kv_cache_block_gather_def.cpp
)

target_sources(optiling PRIVATE
    kv_cache_block_gather_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        kv_cache_block_gather_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
    kv_cache_block_gather_infershape.cpp
    kv_cache_block_gather_graph_plugin.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)
