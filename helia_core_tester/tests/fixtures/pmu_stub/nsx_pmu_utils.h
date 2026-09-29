/* Host stub of the nsx-pmu-armv8m API. */
#ifndef NSX_PMU_UTILS_H
#define NSX_PMU_UTILS_H

#include <stdbool.h>
#include <stdint.h>

#define NSX_PMU_MAX_COUNTERS 8

typedef struct {
    uint32_t apiId;
} nsx_core_api_t;

typedef enum {
    NSX_PMU_EVENT_COUNTER_SIZE_16 = 0,
    NSX_PMU_EVENT_COUNTER_SIZE_32 = 1
} nsx_pmu_event_counter_size_e;

typedef struct {
    bool enabled;
    uint32_t eventId;
    nsx_pmu_event_counter_size_e counterSize;
} nsx_pmu_event_t;

typedef struct {
    bool added;
    uint32_t mapIndex;
    uint32_t counterValue;
} nsx_pmu_counter_t;

typedef struct {
    const nsx_core_api_t *api;
    nsx_pmu_event_t events[NSX_PMU_MAX_COUNTERS];
    nsx_pmu_counter_t counter[NSX_PMU_MAX_COUNTERS];
} nsx_pmu_config_t;

extern const nsx_core_api_t nsx_pmu_V1_0_0;

uint32_t nsx_pmu_init(nsx_pmu_config_t *cfg);
uint32_t nsx_pmu_get_counters(nsx_pmu_config_t *cfg);
void nsx_pmu_event_create(nsx_pmu_event_t *event, uint32_t eventId, nsx_pmu_event_counter_size_e counterSize);
void nsx_pmu_reset_counters(void);
void nsx_pmu_reset_config(nsx_pmu_config_t *cfg);

/* The device header supplies these on target. */
void ARM_PMU_Enable(void);
void ARM_PMU_Disable(void);
void ARM_PMU_CNTR_Disable(uint32_t mask);
void ARM_PMU_Set_CNTR_IRQ_Disable(uint32_t mask);
uint32_t ARM_PMU_Get_CCNTR(void);
uint32_t ARM_PMU_Get_CNTR_OVS(void);

#endif
