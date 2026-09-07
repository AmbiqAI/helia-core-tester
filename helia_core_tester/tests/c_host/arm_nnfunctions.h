/*
 * Host-only stand-in for the ns-cmsis-nn public header, so the generated-test
 * runtime can be compiled and driven by helia_core_tester/tests without a
 * cross toolchain or a kernel checkout. Only the status enum the runtime's
 * validation macros reference is modelled; it is reachable solely through the
 * -I this directory that the host sanity test passes.
 */
#ifndef HELIA_HOST_STUB_ARM_NNFUNCTIONS_H
#define HELIA_HOST_STUB_ARM_NNFUNCTIONS_H

typedef enum
{
    ARM_CMSIS_NN_SUCCESS = 0,
    ARM_CMSIS_NN_ARG_ERROR = -1,
    ARM_CMSIS_NN_NO_IMPL_ERROR = -2,
} arm_cmsis_nn_status;

#endif
