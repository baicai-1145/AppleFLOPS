#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// 发布版仅保留：
// - v4：真实 GEMM（32x32 tile，TK=32，threadgroup padding 降低 bank 冲突，vectorized global loads）
// - peak：compute-amplified 峰值探针（最小写回，逼近矩阵单元理论吞吐）
//
// 约束：
// - FP/BF16 v4：仅支持 N 为 32 的倍数；INT8 v4 可处理任意正 N
// - peak：要求 N 为 8 的倍数（grid=(N/8)x(N/8)）
// - 输入/输出为 row-major 连续存储。
//
// 备注：simdgroup_matrix 指令依赖 Apple GPU 的矩阵加速单元，是测试理论峰值的核心。

enum { TG_THREADS = 128 };

// ----------------
// v4: 32x32 super-tile, TK=32, vectorized global loads + padding
// ----------------

kernel void gemm_fp32_v4(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         constant uint& N [[buffer(3)]],
                         constant uint& inner [[buffer(4)]],
                         uint3 tg_pos [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
  const uint TM = 32;
  const uint TN = 32;
  const uint TK = 32;
  const uint STRIDE_A = TK + 1;
  const uint STRIDE_B = TN + 1;

  const uint row0 = tg_pos.y * TM;
  const uint col0 = tg_pos.x * TN;

  threadgroup float As[TM * STRIDE_A];
  threadgroup float Bs[TK * STRIDE_B];

  const uint sg = tid >> 5;  // 0..3
  const uint sub_r = sg * 8; // 0,8,16,24

  simdgroup_matrix<float, 8, 8> acc0, acc1, acc2, acc3;
  bool first = true;

  const uint vecA = TM * (TK / 4); // 32*8=256
  const uint vecB = TK * (TN / 4); // 32*8=256

  for (uint k0 = 0; k0 < N; k0 += TK) {
    for (uint idx = tid; idx < vecA + vecB; idx += TG_THREADS) {
      if (idx < vecA) {
        const uint r = idx / (TK / 4);
        const uint c4 = (idx % (TK / 4)) * 4;
        const uint off = (row0 + r) * N + (k0 + c4);
        const float4 v = *(const device float4*)(A + off);
        const uint base = r * STRIDE_A + c4;
        As[base + 0] = v.x;
        As[base + 1] = v.y;
        As[base + 2] = v.z;
        As[base + 3] = v.w;
      } else {
        const uint j = idx - vecA;
        const uint r = j / (TN / 4);
        const uint c4 = (j % (TN / 4)) * 4;
        const uint off = (k0 + r) * N + (col0 + c4);
        const float4 v = *(const device float4*)(B + off);
        const uint base = r * STRIDE_B + c4;
        Bs[base + 0] = v.x;
        Bs[base + 1] = v.y;
        Bs[base + 2] = v.z;
        Bs[base + 3] = v.w;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> a0, a1, a2, a3;
    simdgroup_load(a0, As + sub_r * STRIDE_A + 0, STRIDE_A);
    simdgroup_load(a1, As + sub_r * STRIDE_A + 8, STRIDE_A);
    simdgroup_load(a2, As + sub_r * STRIDE_A + 16, STRIDE_A);
    simdgroup_load(a3, As + sub_r * STRIDE_A + 24, STRIDE_A);

    auto step_col = [&](uint col_off, thread simdgroup_matrix<float, 8, 8>& acc) {
      simdgroup_matrix<float, 8, 8> b0, b1, b2, b3;
      simdgroup_load(b0, Bs + 0 * STRIDE_B + col_off, STRIDE_B);
      simdgroup_load(b1, Bs + 8 * STRIDE_B + col_off, STRIDE_B);
      simdgroup_load(b2, Bs + 16 * STRIDE_B + col_off, STRIDE_B);
      simdgroup_load(b3, Bs + 24 * STRIDE_B + col_off, STRIDE_B);

      if (first) {
        simdgroup_multiply(acc, a0, b0);
        simdgroup_multiply_accumulate(acc, a1, b1, acc);
        simdgroup_multiply_accumulate(acc, a2, b2, acc);
        simdgroup_multiply_accumulate(acc, a3, b3, acc);
        for (uint r = 1; r < inner; r++) {
          simdgroup_multiply_accumulate(acc, a0, b0, acc);
          simdgroup_multiply_accumulate(acc, a1, b1, acc);
          simdgroup_multiply_accumulate(acc, a2, b2, acc);
          simdgroup_multiply_accumulate(acc, a3, b3, acc);
        }
      } else {
        simdgroup_multiply_accumulate(acc, a0, b0, acc);
        simdgroup_multiply_accumulate(acc, a1, b1, acc);
        simdgroup_multiply_accumulate(acc, a2, b2, acc);
        simdgroup_multiply_accumulate(acc, a3, b3, acc);
        for (uint r = 1; r < inner; r++) {
          simdgroup_multiply_accumulate(acc, a0, b0, acc);
          simdgroup_multiply_accumulate(acc, a1, b1, acc);
          simdgroup_multiply_accumulate(acc, a2, b2, acc);
          simdgroup_multiply_accumulate(acc, a3, b3, acc);
        }
      }
    };

    step_col(0, acc0);
    step_col(8, acc1);
    step_col(16, acc2);
    step_col(24, acc3);

    first = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  simdgroup_store(acc0, C + (row0 + sub_r) * N + (col0 + 0), N);
  simdgroup_store(acc1, C + (row0 + sub_r) * N + (col0 + 8), N);
  simdgroup_store(acc2, C + (row0 + sub_r) * N + (col0 + 16), N);
  simdgroup_store(acc3, C + (row0 + sub_r) * N + (col0 + 24), N);
}

kernel void gemm_fp16_v4(device const half* A [[buffer(0)]],
                         device const half* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         constant uint& N [[buffer(3)]],
                         constant uint& inner [[buffer(4)]],
                         uint3 tg_pos [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
  const uint TM = 32;
  const uint TN = 32;
  const uint TK = 32;
  const uint STRIDE_A = TK + 1;
  const uint STRIDE_B = TN + 1;

  const uint row0 = tg_pos.y * TM;
  const uint col0 = tg_pos.x * TN;

  threadgroup half As[TM * STRIDE_A];
  threadgroup half Bs[TK * STRIDE_B];

  const uint sg = tid >> 5;
  const uint sub_r = sg * 8;

  simdgroup_matrix<float, 8, 8> acc0, acc1, acc2, acc3;
  bool first = true;

  const uint vecA = TM * (TK / 4);
  const uint vecB = TK * (TN / 4);

  for (uint k0 = 0; k0 < N; k0 += TK) {
    for (uint idx = tid; idx < vecA + vecB; idx += TG_THREADS) {
      if (idx < vecA) {
        const uint r = idx / (TK / 4);
        const uint c4 = (idx % (TK / 4)) * 4;
        const uint off = (row0 + r) * N + (k0 + c4);
        const half4 v = *(const device half4*)(A + off);
        const uint base = r * STRIDE_A + c4;
        As[base + 0] = v.x;
        As[base + 1] = v.y;
        As[base + 2] = v.z;
        As[base + 3] = v.w;
      } else {
        const uint j = idx - vecA;
        const uint r = j / (TN / 4);
        const uint c4 = (j % (TN / 4)) * 4;
        const uint off = (k0 + r) * N + (col0 + c4);
        const half4 v = *(const device half4*)(B + off);
        const uint base = r * STRIDE_B + c4;
        Bs[base + 0] = v.x;
        Bs[base + 1] = v.y;
        Bs[base + 2] = v.z;
        Bs[base + 3] = v.w;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<half, 8, 8> a0, a1, a2, a3;
    simdgroup_load(a0, As + sub_r * STRIDE_A + 0, STRIDE_A);
    simdgroup_load(a1, As + sub_r * STRIDE_A + 8, STRIDE_A);
    simdgroup_load(a2, As + sub_r * STRIDE_A + 16, STRIDE_A);
    simdgroup_load(a3, As + sub_r * STRIDE_A + 24, STRIDE_A);

    auto step_col = [&](uint col_off, thread simdgroup_matrix<float, 8, 8>& acc) {
      simdgroup_matrix<half, 8, 8> b0, b1, b2, b3;
      simdgroup_load(b0, Bs + 0 * STRIDE_B + col_off, STRIDE_B);
      simdgroup_load(b1, Bs + 8 * STRIDE_B + col_off, STRIDE_B);
      simdgroup_load(b2, Bs + 16 * STRIDE_B + col_off, STRIDE_B);
      simdgroup_load(b3, Bs + 24 * STRIDE_B + col_off, STRIDE_B);

      if (first) {
        simdgroup_multiply(acc, a0, b0);
        simdgroup_multiply_accumulate(acc, a1, b1, acc);
        simdgroup_multiply_accumulate(acc, a2, b2, acc);
        simdgroup_multiply_accumulate(acc, a3, b3, acc);
        for (uint r = 1; r < inner; r++) {
          simdgroup_multiply_accumulate(acc, a0, b0, acc);
          simdgroup_multiply_accumulate(acc, a1, b1, acc);
          simdgroup_multiply_accumulate(acc, a2, b2, acc);
          simdgroup_multiply_accumulate(acc, a3, b3, acc);
        }
      } else {
        simdgroup_multiply_accumulate(acc, a0, b0, acc);
        simdgroup_multiply_accumulate(acc, a1, b1, acc);
        simdgroup_multiply_accumulate(acc, a2, b2, acc);
        simdgroup_multiply_accumulate(acc, a3, b3, acc);
        for (uint r = 1; r < inner; r++) {
          simdgroup_multiply_accumulate(acc, a0, b0, acc);
          simdgroup_multiply_accumulate(acc, a1, b1, acc);
          simdgroup_multiply_accumulate(acc, a2, b2, acc);
          simdgroup_multiply_accumulate(acc, a3, b3, acc);
        }
      }
    };

    step_col(0, acc0);
    step_col(8, acc1);
    step_col(16, acc2);
    step_col(24, acc3);

    first = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  simdgroup_store(acc0, C + (row0 + sub_r) * N + (col0 + 0), N);
  simdgroup_store(acc1, C + (row0 + sub_r) * N + (col0 + 8), N);
  simdgroup_store(acc2, C + (row0 + sub_r) * N + (col0 + 16), N);
  simdgroup_store(acc3, C + (row0 + sub_r) * N + (col0 + 24), N);
}

#if defined(__HAVE_BFLOAT__)
kernel void gemm_bf16_v4(device const bfloat* A [[buffer(0)]],
                         device const bfloat* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         constant uint& N [[buffer(3)]],
                         constant uint& inner [[buffer(4)]],
                         uint3 tg_pos [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
  const uint TM = 32;
  const uint TN = 32;
  const uint TK = 32;
  const uint STRIDE_A = TK + 1;
  const uint STRIDE_B = TN + 1;

  const uint row0 = tg_pos.y * TM;
  const uint col0 = tg_pos.x * TN;

  threadgroup bfloat As[TM * STRIDE_A];
  threadgroup bfloat Bs[TK * STRIDE_B];

  const uint sg = tid >> 5;
  const uint sub_r = sg * 8;

  simdgroup_matrix<float, 8, 8> acc0, acc1, acc2, acc3;
  bool first = true;

  const uint vecA = TM * (TK / 4);
  const uint vecB = TK * (TN / 4);

  for (uint k0 = 0; k0 < N; k0 += TK) {
    for (uint idx = tid; idx < vecA + vecB; idx += TG_THREADS) {
      if (idx < vecA) {
        const uint r = idx / (TK / 4);
        const uint c4 = (idx % (TK / 4)) * 4;
        const uint off = (row0 + r) * N + (k0 + c4);
        const bfloat4 v = *(const device bfloat4*)(A + off);
        const uint base = r * STRIDE_A + c4;
        As[base + 0] = v.x;
        As[base + 1] = v.y;
        As[base + 2] = v.z;
        As[base + 3] = v.w;
      } else {
        const uint j = idx - vecA;
        const uint r = j / (TN / 4);
        const uint c4 = (j % (TN / 4)) * 4;
        const uint off = (k0 + r) * N + (col0 + c4);
        const bfloat4 v = *(const device bfloat4*)(B + off);
        const uint base = r * STRIDE_B + c4;
        Bs[base + 0] = v.x;
        Bs[base + 1] = v.y;
        Bs[base + 2] = v.z;
        Bs[base + 3] = v.w;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<bfloat, 8, 8> a0, a1, a2, a3;
    simdgroup_load(a0, As + sub_r * STRIDE_A + 0, STRIDE_A);
    simdgroup_load(a1, As + sub_r * STRIDE_A + 8, STRIDE_A);
    simdgroup_load(a2, As + sub_r * STRIDE_A + 16, STRIDE_A);
    simdgroup_load(a3, As + sub_r * STRIDE_A + 24, STRIDE_A);

    auto step_col = [&](uint col_off, thread simdgroup_matrix<float, 8, 8>& acc) {
      simdgroup_matrix<bfloat, 8, 8> b0, b1, b2, b3;
      simdgroup_load(b0, Bs + 0 * STRIDE_B + col_off, STRIDE_B);
      simdgroup_load(b1, Bs + 8 * STRIDE_B + col_off, STRIDE_B);
      simdgroup_load(b2, Bs + 16 * STRIDE_B + col_off, STRIDE_B);
      simdgroup_load(b3, Bs + 24 * STRIDE_B + col_off, STRIDE_B);

      if (first) {
        simdgroup_multiply(acc, a0, b0);
        simdgroup_multiply_accumulate(acc, a1, b1, acc);
        simdgroup_multiply_accumulate(acc, a2, b2, acc);
        simdgroup_multiply_accumulate(acc, a3, b3, acc);
        for (uint r = 1; r < inner; r++) {
          simdgroup_multiply_accumulate(acc, a0, b0, acc);
          simdgroup_multiply_accumulate(acc, a1, b1, acc);
          simdgroup_multiply_accumulate(acc, a2, b2, acc);
          simdgroup_multiply_accumulate(acc, a3, b3, acc);
        }
      } else {
        simdgroup_multiply_accumulate(acc, a0, b0, acc);
        simdgroup_multiply_accumulate(acc, a1, b1, acc);
        simdgroup_multiply_accumulate(acc, a2, b2, acc);
        simdgroup_multiply_accumulate(acc, a3, b3, acc);
        for (uint r = 1; r < inner; r++) {
          simdgroup_multiply_accumulate(acc, a0, b0, acc);
          simdgroup_multiply_accumulate(acc, a1, b1, acc);
          simdgroup_multiply_accumulate(acc, a2, b2, acc);
          simdgroup_multiply_accumulate(acc, a3, b3, acc);
        }
      }
    };

    step_col(0, acc0);
    step_col(8, acc1);
    step_col(16, acc2);
    step_col(24, acc3);

    first = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  simdgroup_store(acc0, C + (row0 + sub_r) * N + (col0 + 0), N);
  simdgroup_store(acc1, C + (row0 + sub_r) * N + (col0 + 8), N);
  simdgroup_store(acc2, C + (row0 + sub_r) * N + (col0 + 16), N);
  simdgroup_store(acc3, C + (row0 + sub_r) * N + (col0 + 24), N);
}
#endif

kernel void gemm_int8_v4(device const char* A [[buffer(0)]],
                         device const char* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         constant uint& N [[buffer(3)]],
                         constant uint& inner [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const uint col = gid.x;
  const uint row = gid.y;
  if (row >= N || col >= N) return;

  int acc = 0;
  for (uint r = 0; r < inner; r++) {
    for (uint k = 0; k < N; k++) {
      acc += int(A[row * N + k]) * int(B[k * N + col]);
    }
  }
  C[row * N + col] = float(acc);
}

// ----------------
// peak: compute-amplified (minimal writeback)
// - FP/BF16 每个 threadgroup 跑 4 个 simdgroup，反复对固定 8x8 A/B 做 MMA，
//   最后只写回 1 个标量。
// ----------------

kernel void peak_fp32(device float* out [[buffer(0)]],
                      constant uint& inner [[buffer(1)]],
                      uint3 tg_pos [[threadgroup_position_in_grid]],
                      uint3 tgpg [[threadgroups_per_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float As[64];
  threadgroup float Bs[64];
  for (uint i = tid; i < 64; i += TG_THREADS) {
    As[i] = 1.0f + float(i & 7) * 0.001f;
    Bs[i] = 1.0f + float((i >> 3) & 7) * 0.001f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg = tid >> 5;
  simdgroup_matrix<float, 8, 8> a;
  simdgroup_matrix<float, 8, 8> b;
  simdgroup_load(a, As, 8);
  simdgroup_load(b, Bs, 8);

  simdgroup_matrix<float, 8, 8> acc0, acc1, acc2, acc3;
  simdgroup_multiply(acc0, a, b);
  simdgroup_multiply(acc1, a, b);
  simdgroup_multiply(acc2, a, b);
  simdgroup_multiply(acc3, a, b);
  for (uint r = 1; r < inner; r++) {
    simdgroup_multiply_accumulate(acc0, a, b, acc0);
    simdgroup_multiply_accumulate(acc1, a, b, acc1);
    simdgroup_multiply_accumulate(acc2, a, b, acc2);
    simdgroup_multiply_accumulate(acc3, a, b, acc3);
  }

  // 写回时强制使用所有 accumulator，避免编译器 DCE 掉 acc1..acc3 导致 FLOPs 过计数。
  threadgroup float tmp[64 * 4 * 4];
  const uint base = sg * 64 * 4;
  simdgroup_store(acc0, tmp + base + 64 * 0, 8);
  simdgroup_store(acc1, tmp + base + 64 * 1, 8);
  simdgroup_store(acc2, tmp + base + 64 * 2, 8);
  simdgroup_store(acc3, tmp + base + 64 * 3, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const uint idx = tg_pos.y * tgpg.x + tg_pos.x;
    float total = 0.0f;
    for (uint s = 0; s < 4; s++) {
      const uint sbase = s * 64 * 4;
      total += tmp[sbase + 0] + tmp[sbase + 64] + tmp[sbase + 128] + tmp[sbase + 192];
    }
    out[idx] = total;
  }
}

kernel void peak_fp16(device float* out [[buffer(0)]],
                      constant uint& inner [[buffer(1)]],
                      uint3 tg_pos [[threadgroup_position_in_grid]],
                      uint3 tgpg [[threadgroups_per_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half As[64];
  threadgroup half Bs[64];
  for (uint i = tid; i < 64; i += TG_THREADS) {
    const float av = 1.0f + float(i & 7) * 0.001f;
    const float bv = 1.0f + float((i >> 3) & 7) * 0.001f;
    As[i] = half(av);
    Bs[i] = half(bv);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg = tid >> 5;
  simdgroup_matrix<half, 8, 8> a;
  simdgroup_matrix<half, 8, 8> b;
  simdgroup_load(a, As, 8);
  simdgroup_load(b, Bs, 8);

  simdgroup_matrix<float, 8, 8> acc0, acc1, acc2, acc3;
  simdgroup_multiply(acc0, a, b);
  simdgroup_multiply(acc1, a, b);
  simdgroup_multiply(acc2, a, b);
  simdgroup_multiply(acc3, a, b);
  for (uint r = 1; r < inner; r++) {
    simdgroup_multiply_accumulate(acc0, a, b, acc0);
    simdgroup_multiply_accumulate(acc1, a, b, acc1);
    simdgroup_multiply_accumulate(acc2, a, b, acc2);
    simdgroup_multiply_accumulate(acc3, a, b, acc3);
  }

  threadgroup float tmp[64 * 4 * 4];
  const uint base = sg * 64 * 4;
  simdgroup_store(acc0, tmp + base + 64 * 0, 8);
  simdgroup_store(acc1, tmp + base + 64 * 1, 8);
  simdgroup_store(acc2, tmp + base + 64 * 2, 8);
  simdgroup_store(acc3, tmp + base + 64 * 3, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const uint idx = tg_pos.y * tgpg.x + tg_pos.x;
    float total = 0.0f;
    for (uint s = 0; s < 4; s++) {
      const uint sbase = s * 64 * 4;
      total += tmp[sbase + 0] + tmp[sbase + 64] + tmp[sbase + 128] + tmp[sbase + 192];
    }
    out[idx] = total;
  }
}

#if defined(__HAVE_BFLOAT__)
kernel void peak_bf16(device float* out [[buffer(0)]],
                      constant uint& inner [[buffer(1)]],
                      uint3 tg_pos [[threadgroup_position_in_grid]],
                      uint3 tgpg [[threadgroups_per_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat As[64];
  threadgroup bfloat Bs[64];
  for (uint i = tid; i < 64; i += TG_THREADS) {
    const float av = 1.0f + float(i & 7) * 0.001f;
    const float bv = 1.0f + float((i >> 3) & 7) * 0.001f;
    As[i] = (bfloat)av;
    Bs[i] = (bfloat)bv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg = tid >> 5;
  simdgroup_matrix<bfloat, 8, 8> a;
  simdgroup_matrix<bfloat, 8, 8> b;
  simdgroup_load(a, As, 8);
  simdgroup_load(b, Bs, 8);

  simdgroup_matrix<float, 8, 8> acc0, acc1, acc2, acc3;
  simdgroup_multiply(acc0, a, b);
  simdgroup_multiply(acc1, a, b);
  simdgroup_multiply(acc2, a, b);
  simdgroup_multiply(acc3, a, b);
  for (uint r = 1; r < inner; r++) {
    simdgroup_multiply_accumulate(acc0, a, b, acc0);
    simdgroup_multiply_accumulate(acc1, a, b, acc1);
    simdgroup_multiply_accumulate(acc2, a, b, acc2);
    simdgroup_multiply_accumulate(acc3, a, b, acc3);
  }

  threadgroup float tmp[64 * 4 * 4];
  const uint base = sg * 64 * 4;
  simdgroup_store(acc0, tmp + base + 64 * 0, 8);
  simdgroup_store(acc1, tmp + base + 64 * 1, 8);
  simdgroup_store(acc2, tmp + base + 64 * 2, 8);
  simdgroup_store(acc3, tmp + base + 64 * 3, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const uint idx = tg_pos.y * tgpg.x + tg_pos.x;
    float total = 0.0f;
    for (uint s = 0; s < 4; s++) {
      const uint sbase = s * 64 * 4;
      total += tmp[sbase + 0] + tmp[sbase + 64] + tmp[sbase + 128] + tmp[sbase + 192];
    }
    out[idx] = total;
  }
}
#endif

kernel void peak_int8(device float* out [[buffer(0)]],
                      constant uint& inner [[buffer(1)]],
                      uint3 tg_pos [[threadgroup_position_in_grid]],
                      uint3 tgpg [[threadgroups_per_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
  int acc0 = 0;
  int acc1 = 0;
  int acc2 = 0;
  int acc3 = 0;
  for (uint r = 0; r < inner; r++) {
    for (uint k = 0; k < 512; k++) {
      const int a = int((k + tid) & 15) - 8;
      const int b = int(((k >> 4) + tid) & 15) - 8;
      acc0 += a * b;
      acc1 += (a + 1) * b;
      acc2 += a * (b - 1);
      acc3 += (a + 1) * (b - 1);
    }
  }

  threadgroup int tmp[32];
  tmp[tid] = acc0 + acc1 + acc2 + acc3;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    int s = 0;
    for (uint i = 0; i < 32; i++) s += tmp[i];
    const uint idx = tg_pos.y * tgpg.x + tg_pos.x;
    out[idx] = float(s);
  }
}
