#include "../field.cuh"
#include "../memory.cuh"

using namespace ::airbender::field;
using namespace ::airbender::memory;

namespace airbender::ops::hypercube {

DEVICE_FORCEINLINE unsigned swizzle_shared_lane(const unsigned idx) {
  // Mix warp-id bits into lane bits to reduce systematic bank aliasing.
  return idx ^ (idx >> 5);
}

DEVICE_FORCEINLINE unsigned transpose64_conflict_free_idx(const unsigned row6, const unsigned col6) {
  // 64x64 bijection split into:
  //   major = (row << 1) | col_hi
  //   bank  = row_lo ^ col_lo ^ (col_hi << 1)
  //
  // The extra col_hi contribution in bank[1] is enough to keep bank-uniqueness
  // for all three initial-12 access patterns, including the final vec4 gather.
  const unsigned r = row6 & 63u;
  const unsigned c = col6 & 63u;
  const unsigned bank = ((r & 31u) ^ (c & 31u) ^ ((c >> 5) << 1)) & 31u;
  const unsigned major = (r << 1) | (c >> 5);
  return (major << 5) | bank;
}

DEVICE_FORCEINLINE unsigned select_u32(const unsigned a, const unsigned b, const unsigned mask) {
  return (a & ~mask) | (b & mask);
}

DEVICE_FORCEINLINE unsigned noninitial6_log24_smem_idx(const unsigned k, const unsigned p) {
  // 64x32 layout with XOR-swizzled partition index to keep warp-k accesses
  // bank-unique during the 6-round warp-local compute.
  return (k << 5) | (p ^ (k & 31u));
}

template <ld_modifier MODIFIER>
DEVICE_FORCEINLINE uint4 load_u4_mod(const bf *__restrict__ ptr, const unsigned row_base) {
  return load<uint4, MODIFIER>(reinterpret_cast<const uint4 *>(ptr + row_base));
}

template <st_modifier MODIFIER>
DEVICE_FORCEINLINE void store_u4_mod(bf *__restrict__ ptr, const unsigned row_base, const uint4 packed) {
  store<uint4, MODIFIER>(reinterpret_cast<uint4 *>(ptr + row_base), packed);
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

DEVICE_FORCEINLINE void apply_pair_major_high_update_from_peers(
    bf (&vals)[4], const unsigned long long peer01, const unsigned long long peer23) {
  const unsigned peer0 = static_cast<unsigned>(peer01 & 0xFFFFFFFFull);
  const unsigned peer1 = static_cast<unsigned>(peer01 >> 32);
  const unsigned peer2 = static_cast<unsigned>(peer23 & 0xFFFFFFFFull);
  const unsigned peer3 = static_cast<unsigned>(peer23 >> 32);
  vals[0] = bf::sub(vals[0], bf(peer0));
  vals[1] = bf::sub(vals[1], bf(peer1));
  vals[2] = bf::sub(vals[2], bf(peer2));
  vals[3] = bf::sub(vals[3], bf(peer3));
}

DEVICE_FORCEINLINE void apply_6_rounds_pair_major_4groups(bf (&vals)[4][4], const unsigned lane16) {
  // Stage 0+1 are fully local to each lane's four registers.
#pragma unroll
  for (unsigned g = 0; g < 4; g++) {
    vals[g][1] = bf::sub(vals[g][1], vals[g][0]);
    vals[g][3] = bf::sub(vals[g][3], vals[g][2]);
    vals[g][2] = bf::sub(vals[g][2], vals[g][0]);
    vals[g][3] = bf::sub(vals[g][3], vals[g][1]);
  }

#pragma unroll
  for (unsigned bit = 0; bit < 4; bit++) {
    const unsigned lane_bit = 1u << bit;
    const bool high_lane = (lane16 & lane_bit) != 0u;

    const unsigned long long pair01_g0 =
        (static_cast<unsigned long long>(vals[0][1].limb) << 32) | static_cast<unsigned long long>(vals[0][0].limb);
    const unsigned long long pair23_g0 =
        (static_cast<unsigned long long>(vals[0][3].limb) << 32) | static_cast<unsigned long long>(vals[0][2].limb);
    const unsigned long long peer01_g0 = __shfl_xor_sync(0xFFFFFFFFu, pair01_g0, lane_bit, 16);
    const unsigned long long peer23_g0 = __shfl_xor_sync(0xFFFFFFFFu, pair23_g0, lane_bit, 16);

    const unsigned long long pair01_g1 =
        (static_cast<unsigned long long>(vals[1][1].limb) << 32) | static_cast<unsigned long long>(vals[1][0].limb);
    const unsigned long long pair23_g1 =
        (static_cast<unsigned long long>(vals[1][3].limb) << 32) | static_cast<unsigned long long>(vals[1][2].limb);
    const unsigned long long peer01_g1 = __shfl_xor_sync(0xFFFFFFFFu, pair01_g1, lane_bit, 16);
    const unsigned long long peer23_g1 = __shfl_xor_sync(0xFFFFFFFFu, pair23_g1, lane_bit, 16);

    if (high_lane) {
      apply_pair_major_high_update_from_peers(vals[0], peer01_g0, peer23_g0);
    }

    const unsigned long long pair01_g2 =
        (static_cast<unsigned long long>(vals[2][1].limb) << 32) | static_cast<unsigned long long>(vals[2][0].limb);
    const unsigned long long pair23_g2 =
        (static_cast<unsigned long long>(vals[2][3].limb) << 32) | static_cast<unsigned long long>(vals[2][2].limb);
    const unsigned long long peer01_g2 = __shfl_xor_sync(0xFFFFFFFFu, pair01_g2, lane_bit, 16);
    const unsigned long long peer23_g2 = __shfl_xor_sync(0xFFFFFFFFu, pair23_g2, lane_bit, 16);

    if (high_lane) {
      apply_pair_major_high_update_from_peers(vals[1], peer01_g1, peer23_g1);
    }

    const unsigned long long pair01_g3 =
        (static_cast<unsigned long long>(vals[3][1].limb) << 32) | static_cast<unsigned long long>(vals[3][0].limb);
    const unsigned long long pair23_g3 =
        (static_cast<unsigned long long>(vals[3][3].limb) << 32) | static_cast<unsigned long long>(vals[3][2].limb);
    const unsigned long long peer01_g3 = __shfl_xor_sync(0xFFFFFFFFu, pair01_g3, lane_bit, 16);
    const unsigned long long peer23_g3 = __shfl_xor_sync(0xFFFFFFFFu, pair23_g3, lane_bit, 16);

    if (high_lane) {
      apply_pair_major_high_update_from_peers(vals[2], peer01_g2, peer23_g2);
      apply_pair_major_high_update_from_peers(vals[3], peer01_g3, peer23_g3);
    }
  }
}

DEVICE_FORCEINLINE void apply_5_rounds_warp64_pair_branchless_quad(
    bf &a0,
    bf &a1,
    bf &b0,
    bf &b1,
    bf &c0,
    bf &c1,
    bf &d0,
    bf &d1,
    const unsigned lane32) {
#pragma unroll
  for (unsigned bit = 0; bit < 5; bit++) {
    const unsigned mask = 1u << bit;
    const unsigned lane_bit = (lane32 >> bit) & 1u;
    const unsigned lane_mask = 0u - lane_bit;

    const unsigned a0_self = a0.limb;
    const unsigned a1_self = a1.limb;
    const unsigned b0_self = b0.limb;
    const unsigned b1_self = b1.limb;
    const unsigned c0_self = c0.limb;
    const unsigned c1_self = c1.limb;
    const unsigned d0_self = d0.limb;
    const unsigned d1_self = d1.limb;

    const unsigned a0_peer = __shfl_xor_sync(0xFFFFFFFFu, a0_self, mask, 32);
    const unsigned a1_peer = __shfl_xor_sync(0xFFFFFFFFu, a1_self, mask, 32);
    const unsigned b0_peer = __shfl_xor_sync(0xFFFFFFFFu, b0_self, mask, 32);
    const unsigned b1_peer = __shfl_xor_sync(0xFFFFFFFFu, b1_self, mask, 32);
    const unsigned c0_peer = __shfl_xor_sync(0xFFFFFFFFu, c0_self, mask, 32);
    const unsigned c1_peer = __shfl_xor_sync(0xFFFFFFFFu, c1_self, mask, 32);
    const unsigned d0_peer = __shfl_xor_sync(0xFFFFFFFFu, d0_self, mask, 32);
    const unsigned d1_peer = __shfl_xor_sync(0xFFFFFFFFu, d1_self, mask, 32);

    const unsigned a0_lo = select_u32(a0_self, a0_peer, lane_mask);
    const unsigned a0_hi_pre = select_u32(a0_peer, a0_self, lane_mask);
    const unsigned a0_hi = bf::sub(bf(a0_hi_pre), bf(a0_lo)).limb;
    a0 = bf(select_u32(a0_lo, a0_hi, lane_mask));

    const unsigned a1_lo = select_u32(a1_self, a1_peer, lane_mask);
    const unsigned a1_hi_pre = select_u32(a1_peer, a1_self, lane_mask);
    const unsigned a1_hi = bf::sub(bf(a1_hi_pre), bf(a1_lo)).limb;
    a1 = bf(select_u32(a1_lo, a1_hi, lane_mask));

    const unsigned b0_lo = select_u32(b0_self, b0_peer, lane_mask);
    const unsigned b0_hi_pre = select_u32(b0_peer, b0_self, lane_mask);
    const unsigned b0_hi = bf::sub(bf(b0_hi_pre), bf(b0_lo)).limb;
    b0 = bf(select_u32(b0_lo, b0_hi, lane_mask));

    const unsigned b1_lo = select_u32(b1_self, b1_peer, lane_mask);
    const unsigned b1_hi_pre = select_u32(b1_peer, b1_self, lane_mask);
    const unsigned b1_hi = bf::sub(bf(b1_hi_pre), bf(b1_lo)).limb;
    b1 = bf(select_u32(b1_lo, b1_hi, lane_mask));

    const unsigned c0_lo = select_u32(c0_self, c0_peer, lane_mask);
    const unsigned c0_hi_pre = select_u32(c0_peer, c0_self, lane_mask);
    const unsigned c0_hi = bf::sub(bf(c0_hi_pre), bf(c0_lo)).limb;
    c0 = bf(select_u32(c0_lo, c0_hi, lane_mask));

    const unsigned c1_lo = select_u32(c1_self, c1_peer, lane_mask);
    const unsigned c1_hi_pre = select_u32(c1_peer, c1_self, lane_mask);
    const unsigned c1_hi = bf::sub(bf(c1_hi_pre), bf(c1_lo)).limb;
    c1 = bf(select_u32(c1_lo, c1_hi, lane_mask));

    const unsigned d0_lo = select_u32(d0_self, d0_peer, lane_mask);
    const unsigned d0_hi_pre = select_u32(d0_peer, d0_self, lane_mask);
    const unsigned d0_hi = bf::sub(bf(d0_hi_pre), bf(d0_lo)).limb;
    d0 = bf(select_u32(d0_lo, d0_hi, lane_mask));

    const unsigned d1_lo = select_u32(d1_self, d1_peer, lane_mask);
    const unsigned d1_hi_pre = select_u32(d1_peer, d1_self, lane_mask);
    const unsigned d1_hi = bf::sub(bf(d1_hi_pre), bf(d1_lo)).limb;
    d1 = bf(select_u32(d1_lo, d1_hi, lane_mask));
  }
}

DEVICE_FORCEINLINE void apply_5_rounds_warp64_pair_branchless(bf &lo, bf &hi, const unsigned lane32) {
#pragma unroll
  for (unsigned bit = 0; bit < 5; bit++) {
    const unsigned mask = 1u << bit;
    const unsigned lane_bit = (lane32 >> bit) & 1u;
    const unsigned lane_mask = 0u - lane_bit;

    const unsigned lo_self = lo.limb;
    const unsigned hi_self = hi.limb;
    const unsigned lo_peer = __shfl_xor_sync(0xFFFFFFFFu, lo_self, mask, 32);
    const unsigned hi_peer = __shfl_xor_sync(0xFFFFFFFFu, hi_self, mask, 32);

    const unsigned lo_lo = select_u32(lo_self, lo_peer, lane_mask);
    const unsigned lo_hi_pre = select_u32(lo_peer, lo_self, lane_mask);
    const unsigned lo_hi = bf::sub(bf(lo_hi_pre), bf(lo_lo)).limb;
    lo = bf(select_u32(lo_lo, lo_hi, lane_mask));

    const unsigned hi_lo = select_u32(hi_self, hi_peer, lane_mask);
    const unsigned hi_hi_pre = select_u32(hi_peer, hi_self, lane_mask);
    const unsigned hi_hi = bf::sub(bf(hi_hi_pre), bf(hi_lo)).limb;
    hi = bf(select_u32(hi_lo, hi_hi, lane_mask));
  }
}

template <unsigned THREADS = 256>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_initial12_log24(const bf *src,
                                                                            bf *dst,
                                                                            const bool use_cg_loads) {
  static_assert(THREADS == 256);
  constexpr unsigned SUB_SIZE = 1u << 12;
  constexpr unsigned ROWS = 64;
  constexpr unsigned ROWS_PER_WARP = 8;
  constexpr unsigned VEC_ITERS = 4;

  if (blockIdx.y != 0u) {
    return;
  }

  __shared__ __align__(16) bf smem[SUB_SIZE];

  const unsigned tid = threadIdx.x;
  const unsigned warp = tid >> 5;
  const unsigned lane = tid & 31u;
  const unsigned subproblem = blockIdx.x;
  const unsigned block_base = subproblem << 12;

#pragma unroll
  for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
    const unsigned vec = tid + (iter << 8); // iter * 256
    const unsigned row_base = block_base + (vec << 2);
    const uint4 packed = use_cg_loads ? load_u4_mod<ld_modifier::cg>(src, row_base) : load_u4_mod<ld_modifier::cs>(src, row_base);
    reinterpret_cast<uint4 *>(smem + (vec << 2))[0] = packed;
  }

  __syncthreads();

  // Phase 1: 6 rounds across lo-dimension (64-point rows), then write transposed.
  bf phase1_lo[ROWS_PER_WARP];
  bf phase1_hi[ROWS_PER_WARP];
#pragma unroll
  for (unsigned row_local = 0; row_local < ROWS_PER_WARP; row_local++) {
    const unsigned hi6 = warp * ROWS_PER_WARP + row_local;
    const unsigned row_base = hi6 << 6;
    bf lo = smem[row_base + lane];
    bf hi = smem[row_base + 32u + lane];
    apply_5_rounds_warp64_pair_branchless(lo, hi, lane);
    hi = bf::sub(hi, lo);
    phase1_lo[row_local] = lo;
    phase1_hi[row_local] = hi;
  }

  // Ensure all row-major reads are completed before any warp starts writing
  // the transposed layout into the same shared tile.
  __syncthreads();

#pragma unroll
  for (unsigned row_local = 0; row_local < ROWS_PER_WARP; row_local++) {
    const unsigned hi6 = warp * ROWS_PER_WARP + row_local;
    smem[transpose64_conflict_free_idx(lane, hi6)] = phase1_lo[row_local];
    smem[transpose64_conflict_free_idx(lane + 32u, hi6)] = phase1_hi[row_local];
  }

  __syncthreads();

  // Phase 2: 6 rounds across former hi-dimension (now rows after transpose),
  // then transpose back into canonical layout.
  bf phase2_lo[ROWS_PER_WARP];
  bf phase2_hi[ROWS_PER_WARP];
#pragma unroll
  for (unsigned row_local = 0; row_local < ROWS_PER_WARP; row_local++) {
    const unsigned hi6 = warp * ROWS_PER_WARP + row_local;
    bf lo = smem[transpose64_conflict_free_idx(hi6, lane)];
    bf hi = smem[transpose64_conflict_free_idx(hi6, lane + 32u)];
    apply_5_rounds_warp64_pair_branchless(lo, hi, lane);
    hi = bf::sub(hi, lo);
    phase2_lo[row_local] = lo;
    phase2_hi[row_local] = hi;
  }

  // Ensure all transposed reads are completed before back-transpose writes.
  __syncthreads();

#pragma unroll
  for (unsigned row_local = 0; row_local < ROWS_PER_WARP; row_local++) {
    const unsigned hi6 = warp * ROWS_PER_WARP + row_local;
    smem[transpose64_conflict_free_idx(lane, hi6)] = phase2_lo[row_local];
    smem[transpose64_conflict_free_idx(lane + 32u, hi6)] = phase2_hi[row_local];
  }

  __syncthreads();

#pragma unroll
  for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
    const unsigned vec = tid + (iter << 8); // iter * 256
    const unsigned flat = vec << 2;
    const unsigned hi6 = flat >> 6;
    const unsigned lo_base = flat & 63u;
    const uint4 packed = uint4{
        smem[transpose64_conflict_free_idx(hi6, lo_base + 0u)].limb,
        smem[transpose64_conflict_free_idx(hi6, lo_base + 1u)].limb,
        smem[transpose64_conflict_free_idx(hi6, lo_base + 2u)].limb,
        smem[transpose64_conflict_free_idx(hi6, lo_base + 3u)].limb,
    };
    const unsigned row_base = block_base + flat;
    store_u4_mod<st_modifier::cg>(dst, row_base, packed);
  }
}

template <unsigned THREADS = 256>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial6_log24(const bf *__restrict__ src,
                                                                               bf *__restrict__ dst,
                                                                               const bool use_cg_loads,
                                                                               const unsigned start_stage) {
  static_assert(THREADS == 256);
  constexpr unsigned ROUNDS = 6;
  constexpr unsigned K = 1u << ROUNDS; // 64
  constexpr unsigned P = 32;
  constexpr unsigned LOW_TILE_LOG = 5u;
  constexpr unsigned VEC_ITERS = 2;

  if (blockIdx.y != 0u) {
    return;
  }

  __shared__ bf smem[K * P]; // 64 * 32 = 2048 BF values (8KB)

  const unsigned tid = threadIdx.x;
  const unsigned stride = 1u << start_stage;
  const unsigned low_tiles = stride >> LOW_TILE_LOG;
  const unsigned tile = blockIdx.x;
  const unsigned high = tile / low_tiles;
  const unsigned low_tile_id = tile - high * low_tiles;
  const unsigned low_base = low_tile_id << LOW_TILE_LOG;
  const unsigned block_base = high << (start_stage + ROUNDS);

#pragma unroll
  for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
    const unsigned vec = tid + (iter << 8); // iter * 256
    const unsigned k = vec >> 3;
    const unsigned p4 = vec & 7u;
    const unsigned p0 = p4 << 2;
    const unsigned row_base = block_base + (k << start_stage) + low_base + p0;
    const uint4 packed = load_u4_select(src, row_base, use_cg_loads);

    smem[noninitial6_log24_smem_idx(k, p0 + 0u)] = bf(packed.x);
    smem[noninitial6_log24_smem_idx(k, p0 + 1u)] = bf(packed.y);
    smem[noninitial6_log24_smem_idx(k, p0 + 2u)] = bf(packed.z);
    smem[noninitial6_log24_smem_idx(k, p0 + 3u)] = bf(packed.w);
  }

  __syncthreads();

  const unsigned warp = tid >> 5;
  const unsigned lane = tid & 31u;
  const unsigned p0 = warp + 0u;
  const unsigned p1 = warp + 8u;
  const unsigned p2 = warp + 16u;
  const unsigned p3 = warp + 24u;
  const unsigned idx00 = noninitial6_log24_smem_idx(lane + 0u, p0);
  const unsigned idx01 = noninitial6_log24_smem_idx(lane + 32u, p0);
  const unsigned idx10 = noninitial6_log24_smem_idx(lane + 0u, p1);
  const unsigned idx11 = noninitial6_log24_smem_idx(lane + 32u, p1);
  const unsigned idx20 = noninitial6_log24_smem_idx(lane + 0u, p2);
  const unsigned idx21 = noninitial6_log24_smem_idx(lane + 32u, p2);
  const unsigned idx30 = noninitial6_log24_smem_idx(lane + 0u, p3);
  const unsigned idx31 = noninitial6_log24_smem_idx(lane + 32u, p3);

  bf v00 = smem[idx00];
  bf v01 = smem[idx01];
  bf v10 = smem[idx10];
  bf v11 = smem[idx11];
  bf v20 = smem[idx20];
  bf v21 = smem[idx21];
  bf v30 = smem[idx30];
  bf v31 = smem[idx31];

  apply_5_rounds_warp64_pair_branchless_quad(v00, v01, v10, v11, v20, v21, v30, v31, lane);
  v01 = bf::sub(v01, v00);
  v11 = bf::sub(v11, v10);
  v21 = bf::sub(v21, v20);
  v31 = bf::sub(v31, v30);

  smem[idx00] = v00;
  smem[idx01] = v01;
  smem[idx10] = v10;
  smem[idx11] = v11;
  smem[idx20] = v20;
  smem[idx21] = v21;
  smem[idx30] = v30;
  smem[idx31] = v31;

  __syncthreads();

#pragma unroll
  for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
    const unsigned vec = tid + (iter << 8);
    const unsigned k = vec >> 3;
    const unsigned p4 = vec & 7u;
    const unsigned p0 = p4 << 2;
    const unsigned row_base = block_base + (k << start_stage) + low_base + p0;

    const uint4 packed = uint4{
        smem[noninitial6_log24_smem_idx(k, p0 + 0u)].limb,
        smem[noninitial6_log24_smem_idx(k, p0 + 1u)].limb,
        smem[noninitial6_log24_smem_idx(k, p0 + 2u)].limb,
        smem[noninitial6_log24_smem_idx(k, p0 + 3u)].limb,
    };
    store_u4_mod<st_modifier::cg>(dst, row_base, packed);
  }
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

EXTERN __launch_bounds__(256, 5) __global__ void ab_h2m_bitrev_bf_initial_12_kernel(const bf *src,
                                                                                       bf *dst,
                                                                                       const unsigned use_cg_loads,
                                                                                       const unsigned start_stage,
                                                                                       const unsigned log_rows) {
  (void)start_stage;
  (void)log_rows;
  hypercube_evals_into_coeffs_bitrev_initial12_log24<256>(src, dst, use_cg_loads != 0u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial_6_log24_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned use_cg_loads,
    const unsigned start_stage,
    const unsigned log_rows) {
  if (log_rows != 24u || (start_stage + 6u) > log_rows) {
    return;
  }
  hypercube_evals_into_coeffs_bitrev_noninitial6_log24<256>(src, dst, use_cg_loads != 0u, start_stage);
}

// Noninitial kernels (start at big stride)
H2M_NONINITIAL_KERNEL(6);
H2M_NONINITIAL_KERNEL(7);
H2M_NONINITIAL_KERNEL(8);
H2M_NONINITIAL_128_KERNEL(7);

} // namespace airbender::ops::hypercube
