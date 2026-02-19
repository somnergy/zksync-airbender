#include "../field.cuh"
#include "../memory.cuh"

using namespace ::airbender::field;
using namespace ::airbender::memory;

namespace airbender::ops::hypercube {

template <unsigned ROUNDS> DEVICE_FORCEINLINE unsigned transpose_old_to_new(const unsigned old_tid) {
  if constexpr (ROUNDS <= 5) {
    return old_tid;
  } else {
    constexpr unsigned CROSS = ROUNDS - 5;
    constexpr unsigned WARPS = 1u << CROSS;
    const unsigned lane = old_tid & 31u;
    const unsigned warp = old_tid >> 5;
    return lane * WARPS + warp;
  }
}

template <unsigned ROUNDS> DEVICE_FORCEINLINE unsigned transpose_new_to_old(const unsigned new_tid) {
  if constexpr (ROUNDS <= 5) {
    return new_tid;
  } else {
    constexpr unsigned CROSS = ROUNDS - 5;
    constexpr unsigned WARPS = 1u << CROSS;
    // Inverse of new_tid = old_lane * WARPS + old_warp.
    const unsigned old_warp = new_tid & (WARPS - 1u);
    const unsigned old_lane = new_tid >> CROSS;
    return (old_warp << 5) | old_lane;
  }
}

template <unsigned ROUNDS> DEVICE_FORCEINLINE unsigned transpose_old_to_new_swizzled(const unsigned old_tid) {
  if constexpr (ROUNDS <= 5) {
    return old_tid;
  } else {
    constexpr unsigned CROSS = ROUNDS - 5;
    constexpr unsigned WARPS = 1u << CROSS;
    const unsigned old_lane = old_tid & 31u;
    const unsigned old_warp = old_tid >> 5;
    const unsigned warped = old_warp ^ (old_lane & (WARPS - 1u));
    return old_lane * WARPS + warped;
  }
}

template <unsigned ROUNDS> DEVICE_FORCEINLINE unsigned logical_new_to_swizzled_idx(const unsigned new_tid) {
  if constexpr (ROUNDS <= 5) {
    return new_tid;
  } else {
    constexpr unsigned CROSS = ROUNDS - 5;
    constexpr unsigned WARPS = 1u << CROSS;
    const unsigned old_warp = new_tid & (WARPS - 1u);
    const unsigned old_lane = new_tid >> CROSS;
    const unsigned warped = old_warp ^ (old_lane & (WARPS - 1u));
    return old_lane * WARPS + warped;
  }
}

template <unsigned ROUNDS> DEVICE_FORCEINLINE unsigned reorder_write_idx(const unsigned new_tid) {
  if constexpr (ROUNDS <= 5) {
    return new_tid;
  } else {
    const unsigned old_tid = transpose_new_to_old<ROUNDS>(new_tid);
    return (old_tid & ~31u) | ((old_tid & 31u) ^ (new_tid & 31u));
  }
}

template <unsigned ROUNDS> DEVICE_FORCEINLINE unsigned reorder_read_idx(const unsigned old_tid) {
  if constexpr (ROUNDS <= 5) {
    return old_tid;
  } else {
    const unsigned new_tid = transpose_old_to_new<ROUNDS>(old_tid);
    return (old_tid & ~31u) | ((old_tid & 31u) ^ (new_tid & 31u));
  }
}

template <ld_modifier MODIFIER>
DEVICE_FORCEINLINE bf load_bf_group4(const bf *__restrict__ ptr, const unsigned row, const unsigned lane_id) {
  const unsigned lane_group = lane_id & ~3u;
  const unsigned lane_sub = lane_id & 3u;
  const unsigned row_base = row - lane_sub;

  uint4 packed{};
  if (lane_sub == 0u) {
    packed = load<uint4, MODIFIER>(reinterpret_cast<const uint4 *>(ptr + row_base));
  }

  const unsigned x = __shfl_sync(0xffffffffu, packed.x, lane_group);
  const unsigned y = __shfl_sync(0xffffffffu, packed.y, lane_group);
  const unsigned z = __shfl_sync(0xffffffffu, packed.z, lane_group);
  const unsigned w = __shfl_sync(0xffffffffu, packed.w, lane_group);

  unsigned limb = x;
  if (lane_sub == 1u) {
    limb = y;
  } else if (lane_sub == 2u) {
    limb = z;
  } else if (lane_sub == 3u) {
    limb = w;
  }
  return bf(limb);
}

DEVICE_FORCEINLINE bf load_bf_group4_select(const bf *__restrict__ ptr, const unsigned row, const unsigned lane_id, const bool use_cg) {
  if (use_cg) {
    return load_bf_group4<ld_modifier::cg>(ptr, row, lane_id);
  }
  return load_bf_group4<ld_modifier::cs>(ptr, row, lane_id);
}

template <st_modifier MODIFIER>
DEVICE_FORCEINLINE void store_bf_group4(bf *__restrict__ ptr, const unsigned row, const unsigned lane_id, const bf value) {
  const unsigned lane_group = lane_id & ~3u;
  const unsigned lane_sub = lane_id & 3u;
  const unsigned row_base = row - lane_sub;

  const unsigned lane_value = value.limb;
  const unsigned x = __shfl_sync(0xffffffffu, lane_value, lane_group + 0u);
  const unsigned y = __shfl_sync(0xffffffffu, lane_value, lane_group + 1u);
  const unsigned z = __shfl_sync(0xffffffffu, lane_value, lane_group + 2u);
  const unsigned w = __shfl_sync(0xffffffffu, lane_value, lane_group + 3u);

  if (lane_sub == 0u) {
    const uint4 packed = uint4{x, y, z, w};
    store<uint4, MODIFIER>(reinterpret_cast<uint4 *>(ptr + row_base), packed);
  }
}

template <unsigned ROUNDS, bool WARP_SHARED_BACKEND, bool INITIAL>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_fused(const bf *__restrict__ src,
                                                                  bf *__restrict__ dst, const bool use_cg_loads,
                                                                  const unsigned start_stage, const unsigned log_rows) {
  constexpr unsigned SUB_SIZE = 1u << ROUNDS;
  constexpr unsigned LOCAL_ROUNDS = ROUNDS < 5 ? ROUNDS : 5;
  constexpr unsigned MAX_THREADS = 256;
  if (blockIdx.y != 0u) {
    return;
  }

  if constexpr (INITIAL && ROUNDS == 8) {
    __shared__ bf smem[MAX_THREADS];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned subproblem = blockIdx.x;
    const unsigned base = subproblem << ROUNDS;
    const unsigned row = base + tid;

    const bf *__restrict__ src_col = src;
    bf *__restrict__ dst_col = dst;

    bf value = load_bf_group4_select(src_col, row, lane, use_cg_loads);

    if constexpr (WARP_SHARED_BACKEND) {
#pragma unroll
      for (unsigned stage = 0; stage < LOCAL_ROUNDS; stage++) {
        smem[tid] = value;
        __syncwarp();
        if ((tid >> stage) & 1u) {
          value = bf::sub(value, smem[tid ^ (1u << stage)]);
        }
        __syncwarp();
      }
    } else {
#pragma unroll
      for (unsigned stage = 0; stage < LOCAL_ROUNDS; stage++) {
        const unsigned partner = __shfl_xor_sync(0xffffffffu, value.limb, 1u << stage);
        if ((tid >> stage) & 1u) {
          value = bf::sub(value, bf(partner));
        }
      }
    }

#pragma unroll
    for (unsigned stage = 5; stage < ROUNDS; stage++) {
      smem[tid] = value;
      __syncthreads();
      if ((tid >> stage) & 1u) {
        value = bf::sub(value, smem[tid ^ (1u << stage)]);
      }
      __syncthreads();
    }

    store_bf_group4<st_modifier::cg>(dst_col, row, lane, value);
    return;
  }

  if constexpr (INITIAL && ROUNDS > 8) {
    constexpr unsigned EXTRA = 1u << (ROUNDS - 8);
    __shared__ bf smem[EXTRA * MAX_THREADS];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned subproblem = blockIdx.x;
    const unsigned base = subproblem << ROUNDS;

    const bf *__restrict__ src_col = src;
    bf *__restrict__ dst_col = dst;

    bf values[EXTRA];

#pragma unroll
    for (unsigned e = 0; e < EXTRA; e++) {
      const unsigned row = base + (e << 8) + tid;
      values[e] = load_bf_group4_select(src_col, row, lane, use_cg_loads);
    }

    if constexpr (WARP_SHARED_BACKEND) {
#pragma unroll
      for (unsigned stage = 0; stage < 5; stage++) {
#pragma unroll
        for (unsigned e = 0; e < EXTRA; e++) {
          smem[e * MAX_THREADS + tid] = values[e];
        }
        __syncwarp();
        if ((tid >> stage) & 1u) {
#pragma unroll
          for (unsigned e = 0; e < EXTRA; e++) {
            values[e] = bf::sub(values[e], smem[e * MAX_THREADS + (tid ^ (1u << stage))]);
          }
        }
        __syncwarp();
      }
    } else {
#pragma unroll
      for (unsigned stage = 0; stage < 5; stage++) {
#pragma unroll
        for (unsigned e = 0; e < EXTRA; e++) {
          const unsigned partner = __shfl_xor_sync(0xffffffffu, values[e].limb, 1u << stage);
          if ((tid >> stage) & 1u) {
            values[e] = bf::sub(values[e], bf(partner));
          }
        }
      }
    }

#pragma unroll
    for (unsigned stage = 5; stage < 8; stage++) {
#pragma unroll
      for (unsigned e = 0; e < EXTRA; e++) {
        smem[e * MAX_THREADS + tid] = values[e];
      }
      __syncthreads();
      if ((tid >> stage) & 1u) {
#pragma unroll
        for (unsigned e = 0; e < EXTRA; e++) {
          values[e] = bf::sub(values[e], smem[e * MAX_THREADS + (tid ^ (1u << stage))]);
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (unsigned stage = 8; stage < ROUNDS; stage++) {
      const unsigned bit = 1u << (stage - 8);
#pragma unroll
      for (unsigned e = 0; e < EXTRA; e++) {
        if (e & bit) {
          values[e] = bf::sub(values[e], values[e ^ bit]);
        }
      }
    }

#pragma unroll
    for (unsigned e = 0; e < EXTRA; e++) {
      const unsigned row = base + (e << 8) + tid;
      store_bf_group4<st_modifier::cg>(dst_col, row, lane, values[e]);
    }
    return;
  }

  if constexpr (!INITIAL) {
    constexpr unsigned K_WARPS = 1u << (ROUNDS - 5);
    constexpr unsigned LOW_TILE = MAX_THREADS / K_WARPS;
    constexpr unsigned K_ITERS = SUB_SIZE / K_WARPS;
    static_assert(K_ITERS == 32);
    static_assert((LOW_TILE % 4u) == 0u);

    __shared__ bf smem[LOW_TILE * SUB_SIZE];

    const unsigned tid = threadIdx.x;
    const unsigned warp_global = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned k_warp = warp_global % K_WARPS;
    const unsigned low_block = warp_global / K_WARPS;
    const unsigned low_local = (low_block << 5) | lane;

    const unsigned stride = 1u << start_stage;
    const unsigned low_tiles = stride / LOW_TILE;
    const unsigned tile = blockIdx.x;
    const unsigned high = tile / low_tiles;
    const unsigned low_base = (tile - high * low_tiles) * LOW_TILE;
    const unsigned low = low_base + low_local;
    const unsigned block_base = high << (start_stage + ROUNDS);

    const bf *__restrict__ src_col = src;
    bf *__restrict__ dst_col = dst;

    for (unsigned t = 0; t < K_ITERS; t++) {
      const unsigned k = k_warp + t * K_WARPS;
      const unsigned row = block_base + low + (k << start_stage);
      const unsigned low_swizzled = low_local ^ (k & 31u);
      smem[k * LOW_TILE + low_swizzled] = load_bf_group4_select(src_col, row, lane, use_cg_loads);
    }

    __syncthreads();

#pragma unroll
    for (unsigned stage = 0; stage < ROUNDS; stage++) {
      for (unsigned t = 0; t < K_ITERS; t++) {
        const unsigned k = k_warp + t * K_WARPS;
        const unsigned self_idx = k * LOW_TILE + (low_local ^ (k & 31u));
        bf value = smem[self_idx];
        if ((k >> stage) & 1u) {
          const unsigned partner = k ^ (1u << stage);
          const unsigned partner_idx = partner * LOW_TILE + (low_local ^ (partner & 31u));
          value = bf::sub(value, smem[partner_idx]);
        }
        smem[self_idx] = value;
      }
      if constexpr (ROUNDS > 0) {
        if (stage + 1u < ROUNDS) {
          __syncthreads();
        }
      }
    }

    for (unsigned t = 0; t < K_ITERS; t++) {
      const unsigned k = k_warp + t * K_WARPS;
      const unsigned row = block_base + low + (k << start_stage);
      const bf value = smem[k * LOW_TILE + (low_local ^ (k & 31u))];
      store_bf_group4<st_modifier::cg>(dst_col, row, lane, value);
    }
    return;
  }

  __shared__ bf smem[MAX_THREADS];

  const unsigned tid = threadIdx.x;
  const unsigned stage0 = INITIAL ? 0u : start_stage;

  const unsigned subproblem = blockIdx.x;
  const unsigned stride = 1u << stage0;
  const unsigned low = INITIAL ? 0u : (subproblem & (stride - 1u));
  const unsigned high = INITIAL ? subproblem : (subproblem >> stage0);
  const unsigned base = (high << (stage0 + ROUNDS)) + low;

  const unsigned n = 1u << log_rows;
  const bool active = tid < SUB_SIZE;
  const unsigned lane = tid & 31u;
  const bf *__restrict__ src_col = src;
  bf *__restrict__ dst_col = dst;

  bf value = bf::ZERO();
  if (active) {
    const unsigned row = base + (tid << stage0);
    if (row < n) {
      value = load_bf_group4_select(src_col, row, lane, use_cg_loads);
    }
  }

  if constexpr (WARP_SHARED_BACKEND) {
#pragma unroll
    for (unsigned stage = 0; stage < LOCAL_ROUNDS; stage++) {
      if (active)
        smem[tid] = value;
      __syncwarp();
      if (active && ((tid >> stage) & 1u)) {
        const unsigned partner = tid ^ (1u << stage);
        value = bf::sub(value, smem[partner]);
      }
      __syncwarp();
    }
    // Warp-private local rounds reuse the shared-memory buffer.
    // Before cross-warp remap (which writes a global permutation into `smem`),
    // all warps must finish local phases to avoid inter-warp races.
    if constexpr (ROUNDS > 5) {
      __syncthreads();
    }
  } else {
#pragma unroll
    for (unsigned stage = 0; stage < LOCAL_ROUNDS; stage++) {
      const unsigned partner = __shfl_xor_sync(0xffffffffu, value.limb, 1u << stage);
      if (active && ((tid >> stage) & 1u)) {
        value = bf::sub(value, bf(partner));
      }
    }
  }

  if constexpr (ROUNDS > 5) {
    constexpr unsigned CROSS_ROUNDS = ROUNDS - 5;

    if (active) {
      const unsigned remapped_tid = transpose_old_to_new_swizzled<ROUNDS>(tid);
      smem[remapped_tid] = value;
    }

    __syncthreads();

    if (active) {
      value = smem[logical_new_to_swizzled_idx<ROUNDS>(tid)];
#pragma unroll
      for (unsigned stage = 0; stage < CROSS_ROUNDS; stage++) {
        const unsigned partner = __shfl_xor_sync(0xffffffffu, value.limb, 1u << stage);
        if ((tid >> stage) & 1u) {
          value = bf::sub(value, bf(partner));
        }
      }

      smem[reorder_write_idx<ROUNDS>(tid)] = value;
    }
    __syncthreads();
    if (active) {
      const unsigned old_tid = tid;
      const bf reordered = smem[reorder_read_idx<ROUNDS>(old_tid)];
      const unsigned row = base + old_tid;
      if (row < n) {
        store_bf_group4<st_modifier::cg>(dst_col, row, lane, reordered);
      }
    }
  } else {
    if (active) {
      const unsigned row = base + (tid << stage0);
      if (row < n) {
        store_bf_group4<st_modifier::cg>(dst_col, row, lane, value);
      }
    }
  }
}

#define H2M_KERNEL(family, rounds, mode, initial_flag, warp_shared_flag)                                                                                \
  EXTERN __launch_bounds__(256, 2) __global__ void ab_h2m_bitrev_bf_##family##_##rounds##_##mode##_kernel(                                              \
      const bf *__restrict__ src, bf *__restrict__ dst, const unsigned use_cg_loads, const unsigned start_stage, const unsigned log_rows) {               \
    hypercube_evals_into_coeffs_bitrev_fused<rounds, warp_shared_flag, initial_flag>(src, dst, use_cg_loads != 0u, start_stage, log_rows);               \
  }

// Initial kernels (start at stride 1)
H2M_KERNEL(initial, 8, shuffle, true, false);
H2M_KERNEL(initial, 9, shuffle, true, false);
H2M_KERNEL(initial, 10, shuffle, true, false);
H2M_KERNEL(initial, 11, shuffle, true, false);
H2M_KERNEL(initial, 12, shuffle, true, false);

H2M_KERNEL(initial, 8, warp_shared, true, true);
H2M_KERNEL(initial, 9, warp_shared, true, true);
H2M_KERNEL(initial, 10, warp_shared, true, true);
H2M_KERNEL(initial, 11, warp_shared, true, true);
H2M_KERNEL(initial, 12, warp_shared, true, true);

// Noninitial kernels (start at big stride)
H2M_KERNEL(noninitial, 6, shuffle, false, false);
H2M_KERNEL(noninitial, 7, shuffle, false, false);
H2M_KERNEL(noninitial, 8, shuffle, false, false);

H2M_KERNEL(noninitial, 6, warp_shared, false, true);
H2M_KERNEL(noninitial, 7, warp_shared, false, true);
H2M_KERNEL(noninitial, 8, warp_shared, false, true);

} // namespace airbender::ops::hypercube
