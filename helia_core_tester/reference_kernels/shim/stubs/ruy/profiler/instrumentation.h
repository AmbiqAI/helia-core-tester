/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stand-in for ruy's profiler instrumentation: the reference kernels only
 * construct a ScopeLabel, which is a no-op outside a ruy profiling build.
 */
#ifndef HCT_STUB_RUY_PROFILER_INSTRUMENTATION_H
#define HCT_STUB_RUY_PROFILER_INSTRUMENTATION_H

namespace ruy {
namespace profiler {

class ScopeLabel
{
  public:
    template <typename... Args> explicit ScopeLabel(Args &&...) {}
};

} // namespace profiler
} // namespace ruy

#endif
