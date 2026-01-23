#include "field.cuh"

namespace airbender::field {

using bf = base_field;
using e2 = ext2_field;

#define OUTER_COUNT 1024
#define INNER_COUNT 64
#define VALUES_COUNT 16

EXTERN __launch_bounds__(1024, 1) __global__ void ab_bf_add_bench_kernel(bf *values, const unsigned count) {
  bf v[VALUES_COUNT] = {};
  bf r[VALUES_COUNT] = {};
  if (count != 0)
    for (int i = 0; i < VALUES_COUNT; i++)
      v[i] = r[i] = values[i];
#pragma unroll 1
  for (int i = 0; i < OUTER_COUNT; i++)
#pragma unroll
    for (int j = 0; j < INNER_COUNT; j++)
#pragma unroll
      for (int k = 0; k < VALUES_COUNT; k++)
        r[k] = bf::add(r[k], v[k]);
#pragma unroll
  for (int i = 0; i < VALUES_COUNT; i++)
    if (r[i].limb == 0)
      return;
#pragma unroll
  for (int i = 0; i < VALUES_COUNT; i++)
    values[i] = r[i];
}

EXTERN __launch_bounds__(1024, 1) __global__ void ab_bf_mul_bench_kernel(bf *values, const unsigned count) {
  bf v[VALUES_COUNT] = {};
  bf r[VALUES_COUNT] = {};
  if (count != 0)
    for (int i = 0; i < VALUES_COUNT; i++)
      v[i] = r[i] = values[i];
#pragma unroll 1
  for (unsigned i = 0; i < OUTER_COUNT; i++)
#pragma unroll
    for (unsigned j = 0; j < INNER_COUNT; j++)
#pragma unroll
      for (unsigned k = 0; k < VALUES_COUNT; k++)
        r[k] = bf::mul(r[k], v[k]);
#pragma unroll
  for (int i = 0; i < VALUES_COUNT; i++)
    if (r[i].limb == 0)
      return;
#pragma unroll
  for (int i = 0; i < VALUES_COUNT; i++)
    values[i] = r[i];
}

#undef OUTER_COUNT
#undef INNER_COUNT
#undef VALUES_COUNT

#define OUTER_COUNT 256
#define INNER_COUNT 16
#define VALUES_COUNT 16

EXTERN __launch_bounds__(1024, 1) __global__ void ab_e2_sqr_bench_kernel(e2 *values, const unsigned count) {
  e2 v[VALUES_COUNT] = {};
  if (count != 0)
    for (int i = 0; i < VALUES_COUNT; i++)
      v[i] = values[i];
#pragma unroll 1
  for (unsigned i = 0; i < OUTER_COUNT; i++)
#pragma unroll
    for (unsigned j = 0; j < INNER_COUNT; j++)
#pragma unroll
      for (unsigned k = 0; k < VALUES_COUNT; k++)
        v[k] = e2::sqr(v[k]);
#pragma unroll
  for (int i = 0; i < VALUES_COUNT; i++)
    if (v[i][0].limb == 0)
      return;
#pragma unroll
  for (int i = 0; i < VALUES_COUNT; i++)
    values[i] = v[i];
}

} // namespace airbender::field
