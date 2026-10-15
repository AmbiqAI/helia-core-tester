/* Host stub of the module's event map. */
#ifndef NSX_PMU_MAP_H
#define NSX_PMU_MAP_H

#include <stdint.h>

typedef struct {
    uint32_t eventId;
} nsx_pmu_map_t;

#define NSX_PMU_MAP_SIZE 3
extern const nsx_pmu_map_t nsx_pmu_map[];

#endif
