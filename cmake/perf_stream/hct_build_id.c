/* Firmware build-id slot, filled in after the link by scripts/patch_build_id.py.
 *
 * The slot is a fixed-size array holding a recognisable marker followed by a
 * zeroed area. The CMake POST_BUILD step locates the marker in the linked
 * image, hashes the *whole* flash image with the id area zeroed, and writes
 * `hct-<sha256[:48]>` into the area in the ELF and .bin in place. Because the
 * hash spans every loadable byte -- the server objects, cmsis-nn, the NSX
 * board/core/perf/startup libraries and whatever the linker script laid out --
 * two images that differ anywhere get different ids, and identical images get
 * the same id regardless of the build dir they came from.
 *
 * Deliberately not `const`: a const array's contents could be folded into a
 * caller by the compiler (strlen() of a known string), which would bake the
 * unpatched marker into code paths instead of reading the patched slot at
 * run time. The initial value lives in the .data load image, i.e. in flash,
 * which is exactly what gets patched.
 */
#include "benchmark_server_catalog.h"

#define HCT_BUILD_ID_MARKER "HCT-BUILD-ID:"
#define HCT_BUILD_ID_SLOT_BYTES 80u

char hct_build_id_slot[HCT_BUILD_ID_SLOT_BYTES] __attribute__((used)) = HCT_BUILD_ID_MARKER;

const char *hct_benchmark_server_build_id(void)
{
    return hct_build_id_slot + (sizeof(HCT_BUILD_ID_MARKER) - 1u);
}
