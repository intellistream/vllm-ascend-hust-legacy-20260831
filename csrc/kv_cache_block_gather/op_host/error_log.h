#ifndef KV_CACHE_BLOCK_GATHER_ERROR_LOG_H
#define KV_CACHE_BLOCK_GATHER_ERROR_LOG_H

#include <cstdio>

#define OP_LOGI(opname, ...)
#define OP_LOGW(opname, ...)              \
    do {                                  \
        (void)(opname);                   \
        printf("[WARN] ");                \
        printf(__VA_ARGS__);              \
        printf("\n");                    \
    } while (0)

#define OP_LOGE_WITHOUT_REPORT(opname, ...) \
    do {                                    \
        (void)(opname);                     \
        printf("[ERRORx] ");                \
        printf(__VA_ARGS__);                \
        printf("\n");                      \
    } while (0)

#define OP_LOGE(opname, ...)              \
    do {                                  \
        (void)(opname);                   \
        printf("[ERROR] ");               \
        printf(__VA_ARGS__);              \
        printf("\n");                    \
    } while (0)

#define OP_LOGD(opname, ...)

namespace optiling {

#define OP_CHECK_IF(cond, log_func, expr) \
    do {                                  \
        if (cond) {                       \
            log_func;                     \
            expr;                         \
        }                                 \
    } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)              \
    do {                                                      \
        if ((ptr) == nullptr) {                               \
            OP_LOGE((context)->GetNodeType(), "%s is null", #ptr); \
            return ge::GRAPH_FAILED;                          \
        }                                                     \
    } while (0)

} // namespace optiling

#endif // KV_CACHE_BLOCK_GATHER_ERROR_LOG_H
