#ifndef HCT_BENCHMARK_SERVER_MPU_H
#define HCT_BENCHMARK_SERVER_MPU_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Make RAM execute-never; true when on. */
bool hct_mpu_protect(void);

#ifdef __cplusplus
}
#endif

#endif
