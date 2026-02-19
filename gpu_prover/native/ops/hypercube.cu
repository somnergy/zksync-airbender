#include "../field.cuh"
#include "../memory.cuh"

using namespace ::airbender::field;
using namespace ::airbender::memory;

namespace airbender::ops::hypercube {

DEVICE_FORCEINLINE unsigned swizzle_shared_lane(const unsigned idx) {
  // Mix warp-id bits into lane bits to reduce systematic bank aliasing.
  return idx ^ (idx >> 5);
}

template <ld_modifier MODIFIER>
DEVICE_FORCEINLINE uint4 load_u4(const bf *__restrict__ ptr, const unsigned row_base) {
  return load<uint4, MODIFIER>(reinterpret_cast<const uint4 *>(ptr + row_base));
}

DEVICE_FORCEINLINE uint4 load_u4_select(const bf *__restrict__ ptr, const unsigned row_base, const bool use_cg) {
  if (use_cg) {
    return load_u4<ld_modifier::cg>(ptr, row_base);
  }
  return load_u4<ld_modifier::cs>(ptr, row_base);
}

template <unsigned ROUNDS, unsigned THREADS = 256>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_initial(const bf *__restrict__ src,
                                                                    bf *__restrict__ dst, const bool use_cg_loads) {
  constexpr unsigned SUB_SIZE = 1u << ROUNDS;
  constexpr unsigned GROUPS = SUB_SIZE >> 2;
  constexpr unsigned MAX_THREADS = THREADS;
  static_assert(MAX_THREADS == 256);
  static_assert((SUB_SIZE % 4) == 0);
  if (blockIdx.y != 0u) {
    return;
  }

  __shared__ bf smem[SUB_SIZE];

  const unsigned tid = threadIdx.x;
  const unsigned subproblem = blockIdx.x;
  const unsigned base = subproblem << ROUNDS;

  for (unsigned group = tid; group < GROUPS; group += MAX_THREADS) {
    const unsigned local_base = group << 2;
    const unsigned row_base = base + local_base;
    const uint4 packed = load_u4_select(src, row_base, use_cg_loads);
    smem[swizzle_shared_lane(local_base + 0)] = bf(packed.x);
    smem[swizzle_shared_lane(local_base + 1)] = bf(packed.y);
    smem[swizzle_shared_lane(local_base + 2)] = bf(packed.z);
    smem[swizzle_shared_lane(local_base + 3)] = bf(packed.w);
  }

  __syncthreads();

#pragma unroll
  for (unsigned stage = 0; stage < ROUNDS; stage++) {
    const unsigned bit = 1u << stage;
    for (unsigned idx = tid; idx < SUB_SIZE; idx += MAX_THREADS) {
      if ((idx & bit) == 0u) {
        continue;
      }
      const unsigned self_idx = swizzle_shared_lane(idx);
      const unsigned partner_idx = swizzle_shared_lane(idx ^ bit);
      smem[self_idx] = bf::sub(smem[self_idx], smem[partner_idx]);
    }
    if (stage + 1u < ROUNDS) {
      __syncthreads();
    }
  }
  __syncthreads();

  for (unsigned group = tid; group < GROUPS; group += MAX_THREADS) {
    const unsigned local_base = group << 2;
    const unsigned row_base = base + local_base;
    const uint4 packed = uint4{
        smem[swizzle_shared_lane(local_base + 0)].limb,
        smem[swizzle_shared_lane(local_base + 1)].limb,
        smem[swizzle_shared_lane(local_base + 2)].limb,
        smem[swizzle_shared_lane(local_base + 3)].limb,
    };
    store<uint4, st_modifier::cg>(reinterpret_cast<uint4 *>(dst + row_base), packed);
  }
}

template <unsigned ROUNDS, unsigned THREADS = 256>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial(const bf *__restrict__ src,
                                                                       bf *__restrict__ dst, const bool use_cg_loads,
                                                                       const unsigned start_stage) {
  constexpr unsigned SUB_SIZE = 1u << ROUNDS;
  constexpr unsigned MAX_THREADS = THREADS;
  if (blockIdx.y != 0u) {
    return;
  }

  // Noninitial kernels process "k" dimension in tiles:
  // each CTA owns LOW_TILE contiguous low indices and walks all k iters.
  constexpr unsigned K_WARPS = 1u << (ROUNDS - 5);
  constexpr unsigned LOW_TILE = MAX_THREADS / K_WARPS;
  static_assert((LOW_TILE % 4u) == 0u);

  constexpr unsigned LOW_GROUPS = LOW_TILE / 4;
  constexpr unsigned GROUPS_PER_K = LOW_GROUPS;
  constexpr unsigned LOW_MASK = LOW_TILE - 1u;
  constexpr unsigned LOW_LOG = (LOW_TILE == 128u) ? 7u : ((LOW_TILE == 64u) ? 6u : 5u);
  constexpr unsigned LOW_GROUP_LOG = (LOW_GROUPS == 32u) ? 5u : ((LOW_GROUPS == 16u) ? 4u : 3u);
  constexpr unsigned K_VEC_STRIDE = MAX_THREADS >> LOW_GROUP_LOG;
  constexpr unsigned K_STAGE_STRIDE = MAX_THREADS >> LOW_LOG;
  static_assert((LOW_TILE & LOW_MASK) == 0u);
  static_assert((1u << LOW_LOG) == LOW_TILE);
  static_assert((1u << LOW_GROUP_LOG) == LOW_GROUPS);
  static_assert(K_VEC_STRIDE > 0u);
  static_assert(K_STAGE_STRIDE > 0u);

  __shared__ bf smem[LOW_TILE * SUB_SIZE];

  const unsigned tid = threadIdx.x;
  const unsigned stride = 1u << start_stage;
  const unsigned low_tiles = stride / LOW_TILE;
  const unsigned tile = blockIdx.x;
  const unsigned high = tile / low_tiles;
  const unsigned low_base = (tile - high * low_tiles) * LOW_TILE;
  const unsigned block_base = high << (start_stage + ROUNDS);

  const unsigned low_group0 = tid & (LOW_GROUPS - 1u);
  const unsigned k_vec0 = tid >> LOW_GROUP_LOG;
  for (unsigned k = k_vec0; k < SUB_SIZE; k += K_VEC_STRIDE) {
    const unsigned low_group = low_group0;
    const unsigned low = low_base + (low_group << 2);
    const unsigned row_base = block_base + low + (k << start_stage);
    const uint4 packed = load_u4_select(src, row_base, use_cg_loads);
    const unsigned low_local = low_group << 2;
    const unsigned k_swizzle = k & LOW_MASK;
    const unsigned smem_base = k * LOW_TILE;
    smem[smem_base + ((low_local + 0) ^ k_swizzle)] = bf(packed.x);
    smem[smem_base + ((low_local + 1) ^ k_swizzle)] = bf(packed.y);
    smem[smem_base + ((low_local + 2) ^ k_swizzle)] = bf(packed.z);
    smem[smem_base + ((low_local + 3) ^ k_swizzle)] = bf(packed.w);
  }

  __syncthreads();

  const unsigned low_off = tid & LOW_MASK;
  const unsigned k_stage0 = tid >> LOW_LOG;
#pragma unroll
  for (unsigned stage = 0; stage < ROUNDS; stage++) {
    const unsigned bit = 1u << stage;
    for (unsigned k = k_stage0; k < SUB_SIZE; k += K_STAGE_STRIDE) {
      if ((k & bit) == 0u) {
        continue;
      }
      const unsigned partner = k ^ bit;
      const unsigned self_idx = k * LOW_TILE + (low_off ^ (k & LOW_MASK));
      const unsigned partner_idx = partner * LOW_TILE + (low_off ^ (partner & LOW_MASK));
      smem[self_idx] = bf::sub(smem[self_idx], smem[partner_idx]);
    }
    if (stage + 1u < ROUNDS) {
      __syncthreads();
    }
  }
  __syncthreads();

  for (unsigned k = k_vec0; k < SUB_SIZE; k += K_VEC_STRIDE) {
    const unsigned low_group = low_group0;
    const unsigned low = low_base + (low_group << 2);
    const unsigned row_base = block_base + low + (k << start_stage);
    const unsigned low_local = low_group << 2;
    const unsigned k_swizzle = k & LOW_MASK;
    const unsigned smem_base = k * LOW_TILE;
    const uint4 packed = uint4{
        smem[smem_base + ((low_local + 0) ^ k_swizzle)].limb,
        smem[smem_base + ((low_local + 1) ^ k_swizzle)].limb,
        smem[smem_base + ((low_local + 2) ^ k_swizzle)].limb,
        smem[smem_base + ((low_local + 3) ^ k_swizzle)].limb,
    };
    store<uint4, st_modifier::cg>(reinterpret_cast<uint4 *>(dst + row_base), packed);
  }
}

#define H2M_INITIAL_KERNEL(rounds)                                                                                                                        \
  EXTERN __launch_bounds__(256, 2) __global__ void ab_h2m_bitrev_bf_initial_##rounds##_kernel(                                                          \
      const bf *__restrict__ src, bf *__restrict__ dst, const unsigned use_cg_loads, const unsigned start_stage, const unsigned log_rows) {               \
    (void)start_stage;                                                                                                                                    \
    (void)log_rows;                                                                                                                                       \
    hypercube_evals_into_coeffs_bitrev_initial<rounds, 256>(src, dst, use_cg_loads != 0u);                                                              \
  }

#define H2M_NONINITIAL_KERNEL(rounds)                                                                                                                     \
  EXTERN __launch_bounds__(256, 2) __global__ void ab_h2m_bitrev_bf_noninitial_##rounds##_kernel(                                                       \
      const bf *__restrict__ src, bf *__restrict__ dst, const unsigned use_cg_loads, const unsigned start_stage, const unsigned log_rows) {               \
    (void)log_rows;                                                                                                                                       \
    hypercube_evals_into_coeffs_bitrev_noninitial<rounds, 256>(src, dst, use_cg_loads != 0u, start_stage);                                              \
  }

#define H2M_NONINITIAL_128_KERNEL(rounds)                                                                                                                 \
  EXTERN __launch_bounds__(128, 2) __global__ void ab_h2m_bitrev_bf_noninitial_##rounds##_128_kernel(                                                   \
      const bf *__restrict__ src, bf *__restrict__ dst, const unsigned use_cg_loads, const unsigned start_stage, const unsigned log_rows) {               \
    (void)log_rows;                                                                                                                                       \
    hypercube_evals_into_coeffs_bitrev_noninitial<rounds, 128>(src, dst, use_cg_loads != 0u, start_stage);                                              \
  }

// Initial kernels (start at stride 1)
H2M_INITIAL_KERNEL(8);
H2M_INITIAL_KERNEL(9);
H2M_INITIAL_KERNEL(10);
H2M_INITIAL_KERNEL(11);
H2M_INITIAL_KERNEL(12);

// Noninitial kernels (start at big stride)
H2M_NONINITIAL_KERNEL(6);
H2M_NONINITIAL_KERNEL(7);
H2M_NONINITIAL_KERNEL(8);
H2M_NONINITIAL_128_KERNEL(7);

} // namespace airbender::ops::hypercube
