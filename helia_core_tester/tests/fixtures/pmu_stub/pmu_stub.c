/* Host PMU stub: fixed, checkable readings. */
#include <string.h>

#include "nsx_pmu_map.h"
#include "nsx_pmu_utils.h"

#define STUB_CCNTR 1234u
/* CCNTR, slot 0 and slot 3 overflowed. */
#define STUB_OVS 0x80000009u

const nsx_core_api_t nsx_pmu_V1_0_0 = {0xCA000Bu};
const nsx_pmu_map_t nsx_pmu_map[NSX_PMU_MAP_SIZE] = {{0x0008u}, {0x0023u}, {0x0200u}};

static uint32_t s_ccntr;
static uint32_t s_ovs;

void nsx_pmu_reset_config(nsx_pmu_config_t *cfg) { memset(cfg, 0, sizeof(*cfg)); }

void nsx_pmu_event_create(nsx_pmu_event_t *event, uint32_t eventId, nsx_pmu_event_counter_size_e counterSize)
{
    event->enabled = true;
    event->eventId = eventId;
    event->counterSize = counterSize;
}

uint32_t nsx_pmu_init(nsx_pmu_config_t *cfg) { return cfg->api == &nsx_pmu_V1_0_0 ? 0u : 1u; }

void nsx_pmu_reset_counters(void)
{
    s_ccntr = 0u;
    s_ovs = 0u;
}

/* Value encodes id and counter size. */
uint32_t nsx_pmu_get_counters(nsx_pmu_config_t *cfg)
{
    uint32_t index;
    for (index = 0u; index < NSX_PMU_MAX_COUNTERS; ++index)
    {
        cfg->counter[index].counterValue = cfg->events[index].eventId + ((uint32_t)cfg->events[index].counterSize << 16);
    }
    /* The real read resets CCNTR and OVS. */
    nsx_pmu_reset_counters();
    return 0u;
}

/* Starting a sample "runs" it. */
void ARM_PMU_Enable(void)
{
    s_ccntr = STUB_CCNTR;
    s_ovs = STUB_OVS;
}

void ARM_PMU_Disable(void) {}
void ARM_PMU_CNTR_Disable(uint32_t mask) { (void)mask; }
void ARM_PMU_Set_CNTR_IRQ_Disable(uint32_t mask) { (void)mask; }
uint32_t ARM_PMU_Get_CCNTR(void) { return s_ccntr; }
uint32_t ARM_PMU_Get_CNTR_OVS(void) { return s_ovs; }
