#include <Accelerate/Accelerate.h>

#include <array>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <sys/sysctl.h>
#include <unistd.h>
#include <vector>

#include <arm_neon.h>

#include "gpu_bench.h"

namespace {

enum class Precision { NA, FP16, FP32, BF16, INT8 };

struct BenchRow {
  int n = 0;
  std::string unit;
  Precision precision = Precision::NA;
  double seconds = 0.0;
  double tflops = 0.0;
  std::optional<double> watts;
  bool throttling = false;
  std::string note;
};

struct Options {
  enum class Unit { CPU, GPU };
  enum class Mode { AMX, Ref, Both };
  enum class Power { None, Powermetrics };
  enum class GpuKernel { Auto, V4 };
  enum class GpuStorage { Shared, Private };
  enum class GpuWorkload { Gemm, Peak };

  Unit unit = Unit::CPU;
  Mode mode = Mode::Both;
  int n = 1024;
  int warmup = 1;
  int repeats = 5;
  bool verify = true;
  bool precision_set = false;

  std::vector<Precision> precisions;
  std::string gpu_shader_path = "shaders/gemm.metal";
  GpuKernel gpu_kernel = GpuKernel::Auto;
  GpuStorage gpu_storage = GpuStorage::Private;
  int gpu_batch = 1;
  int gpu_inner = 1;
  GpuWorkload gpu_workload = GpuWorkload::Gemm;

  bool sweep = false;
  int sweep_min = 512;
  int sweep_max = 16384;
  int stress = 0;  // 0=off, else run same config N times

  Power power = Power::None;
  int power_every = 5;  // 仅在 --stress 中生效：每 N 次迭代采样一次功耗，避免 powermetrics 过度干扰。
};

[[noreturn]] void die(std::string_view msg) {
  std::cerr << "error: " << msg << "\n";
  std::exit(1);
}

bool parse_int(std::string_view s, int& out) {
  if (s.empty()) return false;
  int sign = 1;
  size_t i = 0;
  if (s[0] == '-') {
    sign = -1;
    i = 1;
  }
  if (i >= s.size()) return false;
  long long v = 0;
  for (; i < s.size(); i++) {
    if (s[i] < '0' || s[i] > '9') return false;
    v = v * 10 + (s[i] - '0');
    if (v > 1'000'000'000LL) return false;
  }
  out = static_cast<int>(v * sign);
  return true;
}

Precision precision_from_gpu(GpuPrecision p) {
  if (p == GpuPrecision::FP16) return Precision::FP16;
  if (p == GpuPrecision::FP32) return Precision::FP32;
  if (p == GpuPrecision::BF16) return Precision::BF16;
  return Precision::INT8;
}

GpuPrecision gpu_precision_from_precision(Precision p) {
  if (p == Precision::FP16) return GpuPrecision::FP16;
  if (p == Precision::FP32) return GpuPrecision::FP32;
  if (p == Precision::BF16) return GpuPrecision::BF16;
  if (p == Precision::INT8) return GpuPrecision::INT8;
  die("invalid GPU precision");
}

int precision_index(Precision p) {
  if (p == Precision::FP16) return 0;
  if (p == Precision::FP32) return 1;
  if (p == Precision::BF16) return 2;
  if (p == Precision::INT8) return 3;
  return 0;
}

std::vector<Precision> parse_precisions(std::string_view v) {
  if (v == "both") return {Precision::FP16, Precision::FP32};
  if (v == "all") return {Precision::FP16, Precision::FP32, Precision::BF16, Precision::INT8};
  if (v == "fp16") return {Precision::FP16};
  if (v == "fp32") return {Precision::FP32};
  if (v == "bf16") return {Precision::BF16};
  if (v == "int8" || v == "i8") return {Precision::INT8};
  die("--precision must be one of: fp16, fp32, bf16, int8, both, all");
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    std::string_view a(argv[i]);
    auto need_value = [&](std::string_view flag) -> std::string_view {
      if (i + 1 >= argc) die(std::string(flag) + " needs a value");
      return std::string_view(argv[++i]);
    };

    if (a == "-h" || a == "--help") {
      std::cout
          << "MTFLOPS (Stage 1/2: CPU AMX + Metal GPU)\n\n"
          << "Usage:\n"
          << "  ./mtflops [--unit cpu|gpu] [--mode amx|ref|both] [--precision fp16|fp32|bf16|int8|both|all]\n"
          << "           [--n N] [--warmup W] [--repeats R] [--verify 0|1]\n"
          << "           [--shader <path>]\n\n"
          << "           [--sweep 0|1] [--sweep-min N] [--sweep-max N]\n"
          << "           [--stress ITERS]\n"
          << "           [--power none|powermetrics] [--power-every K]\n"
          << "           [--kernel auto|v4]\n\n"
          << "           [--gpu-storage shared|private] [--gpu-batch B]\n\n"
          << "           [--gpu-inner I]  (I>1 为 compute-amplified，趋近理论峰值用)\n\n"
          << "           [--gpu-workload gemm|peak]\n\n"
          << "Notes:\n"
          << "  - GEMM 口径：浮点按 2*N^3 计 TFLOPS；INT8 按 2*N^3 计 TOPS。\n"
          << "  - FP32 AMX 路径通过 Accelerate 的 cblas_sgemm 触发；CPU FP16/INT8 优先使用 NEON，BF16 使用 verified typed kernel。\n";
      std::exit(0);
    } else if (a == "--unit") {
      auto v = need_value("--unit");
      if (v == "cpu") opt.unit = Options::Unit::CPU;
      else if (v == "gpu") opt.unit = Options::Unit::GPU;
      else die("--unit must be one of: cpu, gpu");
    } else if (a == "--mode") {
      auto v = need_value("--mode");
      if (v == "amx") opt.mode = Options::Mode::AMX;
      else if (v == "ref") opt.mode = Options::Mode::Ref;
      else if (v == "both") opt.mode = Options::Mode::Both;
      else die("--mode must be one of: amx, ref, both");
    } else if (a == "--precision") {
      auto v = need_value("--precision");
      opt.precisions = parse_precisions(v);
      opt.precision_set = true;
    } else if (a == "--kernel") {
      auto v = need_value("--kernel");
      if (v == "auto") opt.gpu_kernel = Options::GpuKernel::Auto;
      else if (v == "v4") opt.gpu_kernel = Options::GpuKernel::V4;
      else die("--kernel must be one of: auto, v4");
    } else if (a == "--gpu-storage") {
      auto v = need_value("--gpu-storage");
      if (v == "shared") opt.gpu_storage = Options::GpuStorage::Shared;
      else if (v == "private") opt.gpu_storage = Options::GpuStorage::Private;
      else die("--gpu-storage must be one of: shared, private");
    } else if (a == "--gpu-batch") {
      int v = 0;
      if (!parse_int(need_value("--gpu-batch"), v) || v <= 0) die("--gpu-batch must be > 0");
      opt.gpu_batch = v;
    } else if (a == "--gpu-inner") {
      int v = 0;
      if (!parse_int(need_value("--gpu-inner"), v) || v <= 0) die("--gpu-inner must be > 0");
      opt.gpu_inner = v;
    } else if (a == "--gpu-workload") {
      auto v = need_value("--gpu-workload");
      if (v == "gemm") opt.gpu_workload = Options::GpuWorkload::Gemm;
      else if (v == "peak") opt.gpu_workload = Options::GpuWorkload::Peak;
      else die("--gpu-workload must be one of: gemm, peak");
    } else if (a == "--shader") {
      opt.gpu_shader_path = std::string(need_value("--shader"));
    } else if (a == "--sweep") {
      int v = 0;
      if (!parse_int(need_value("--sweep"), v) || (v != 0 && v != 1)) die("--sweep must be 0 or 1");
      opt.sweep = (v == 1);
    } else if (a == "--sweep-min") {
      int v = 0;
      if (!parse_int(need_value("--sweep-min"), v) || v <= 0) die("--sweep-min must be > 0");
      opt.sweep_min = v;
    } else if (a == "--sweep-max") {
      int v = 0;
      if (!parse_int(need_value("--sweep-max"), v) || v <= 0) die("--sweep-max must be > 0");
      opt.sweep_max = v;
    } else if (a == "--stress") {
      int v = 0;
      if (!parse_int(need_value("--stress"), v) || v < 0) die("--stress must be >= 0");
      opt.stress = v;
    } else if (a == "--power") {
      auto v = need_value("--power");
      if (v == "none") opt.power = Options::Power::None;
      else if (v == "powermetrics") opt.power = Options::Power::Powermetrics;
      else die("--power must be one of: none, powermetrics");
    } else if (a == "--power-every") {
      int v = 0;
      if (!parse_int(need_value("--power-every"), v) || v <= 0) die("--power-every must be > 0");
      opt.power_every = v;
    } else if (a == "--n") {
      int v = 0;
      if (!parse_int(need_value("--n"), v) || v <= 0) die("--n must be a positive integer");
      opt.n = v;
    } else if (a == "--warmup") {
      int v = 0;
      if (!parse_int(need_value("--warmup"), v) || v < 0) die("--warmup must be >= 0");
      opt.warmup = v;
    } else if (a == "--repeats") {
      int v = 0;
      if (!parse_int(need_value("--repeats"), v) || v <= 0) die("--repeats must be a positive integer");
      opt.repeats = v;
    } else if (a == "--verify") {
      int v = 0;
      if (!parse_int(need_value("--verify"), v) || (v != 0 && v != 1)) die("--verify must be 0 or 1");
      opt.verify = (v == 1);
    } else {
      die(std::string("unknown arg: ") + std::string(a));
    }
  }
  if (!opt.precision_set) {
    opt.precisions = (opt.unit == Options::Unit::GPU) ? std::vector<Precision>{Precision::FP16}
                                                      : std::vector<Precision>{Precision::FP32};
  }
  return opt;
}

template <typename T>
T* aligned_alloc_64(size_t count) {
  void* p = nullptr;
  const size_t bytes = count * sizeof(T);
  if (posix_memalign(&p, 64, bytes) != 0 || p == nullptr) die("posix_memalign failed");
  std::memset(p, 0, bytes);
  return static_cast<T*>(p);
}

bool cpu_feature_enabled(const char* name) {
  int v = 0;
  size_t len = sizeof(v);
  return sysctlbyname(name, &v, &len, nullptr, 0) == 0 && v != 0;
}

bool cpu_has_fp16() {
  static const bool yes = cpu_feature_enabled("hw.optional.arm.FEAT_FP16");
  return yes;
}

bool cpu_has_dotprod() {
  static const bool yes = cpu_feature_enabled("hw.optional.arm.FEAT_DotProd");
  return yes;
}

bool cpu_use_bf16_neon() {
  // FEAT_BF16 is present on M4, but the current NEON BF16 dot layout is not
  // trustworthy for row-major GEMM. Keep BF16 on the verified blocked path.
  return false;
}

template <typename T>
void transpose_square(const T* src, T* dst, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      dst[static_cast<size_t>(j) * n + i] = src[static_cast<size_t>(i) * n + j];
    }
  }
}

void fill_matrix(float* m, int n, uint32_t seed) {
  // 固定 seed 的轻量初始化：避免跑分时随机数开销，且保证可复现。
  uint32_t x = seed ? seed : 1u;
  const int nn = n * n;
  for (int i = 0; i < nn; i++) {
    x = x * 1664525u + 1013904223u;
    // [0,1) 的伪随机 float
    m[i] = static_cast<float>((x >> 8) & 0x00FFFFFF) / static_cast<float>(0x01000000);
  }
}

void fill_matrix_fp16(__fp16* m, int n, uint32_t seed) {
  uint32_t x = seed ? seed : 1u;
  const int nn = n * n;
  for (int i = 0; i < nn; i++) {
    x = x * 1664525u + 1013904223u;
    const float f = static_cast<float>((x >> 8) & 0x00FFFFFF) / static_cast<float>(0x01000000);
    m[i] = static_cast<__fp16>(f);
  }
}

uint16_t float_to_bf16_bits(float f) {
  uint32_t u = 0;
  static_assert(sizeof(float) == sizeof(uint32_t));
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t lsb = (u >> 16) & 1u;
  const uint32_t bias = 0x7FFFu + lsb;  // round-to-nearest-even
  return static_cast<uint16_t>((u + bias) >> 16);
}

float bf16_bits_to_float(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

void fill_matrix_bf16_as_float(float* m, int n, uint32_t seed) {
  uint32_t x = seed ? seed : 1u;
  const int nn = n * n;
  for (int i = 0; i < nn; i++) {
    x = x * 1664525u + 1013904223u;
    const float f = static_cast<float>((x >> 8) & 0x00FFFFFF) / static_cast<float>(0x01000000);
    m[i] = bf16_bits_to_float(float_to_bf16_bits(f));
  }
}

void fill_matrix_bf16(uint16_t* m, int n, uint32_t seed) {
  uint32_t x = seed ? seed : 1u;
  const int nn = n * n;
  for (int i = 0; i < nn; i++) {
    x = x * 1664525u + 1013904223u;
    const float f = static_cast<float>((x >> 8) & 0x00FFFFFF) / static_cast<float>(0x01000000);
    m[i] = float_to_bf16_bits(f);
  }
}

void fill_matrix_int8(int8_t* m, int n, uint32_t seed) {
  uint32_t x = seed ? seed : 1u;
  const int nn = n * n;
  for (int i = 0; i < nn; i++) {
    x = x * 1664525u + 1013904223u;
    m[i] = static_cast<int8_t>(((x >> 24) & 0x0F) - 8);
  }
}

void gemm_amx_accelerate(const float* a, const float* b, float* c, int n) {
  // Row-major: C = A * B
  // beta=0 保证不读取 C 的初值，避免额外带宽。
  cblas_sgemm(CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              n,
              n,
              n,
              1.0f,
              a,
              n,
              b,
              n,
              0.0f,
              c,
              n);
}

void gemm_ref_blocked(const float* a, const float* b, float* c, int n) {
  // 朴素三重循环在 N 稍大时会非常慢。这里用最小复杂度的 blocking 提升缓存命中率，
  // 作为“普通 CPU 核心”的对照组（KISS：不引入 NEON/汇编）。
  constexpr int BS = 64;
  std::fill(c, c + static_cast<size_t>(n) * static_cast<size_t>(n), 0.0f);

  for (int ii = 0; ii < n; ii += BS) {
    const int i_end = std::min(ii + BS, n);
    for (int kk = 0; kk < n; kk += BS) {
      const int k_end = std::min(kk + BS, n);
      for (int jj = 0; jj < n; jj += BS) {
        const int j_end = std::min(jj + BS, n);

        for (int i = ii; i < i_end; i++) {
          const float* a_row = a + static_cast<size_t>(i) * n;
          float* c_row = c + static_cast<size_t>(i) * n;
          for (int k = kk; k < k_end; k++) {
            const float a_ik = a_row[k];
            const float* b_row = b + static_cast<size_t>(k) * n;
            for (int j = jj; j < j_end; j++) {
              c_row[j] += a_ik * b_row[j];
            }
          }
        }
      }
    }
  }
}

void gemm_fp16_blocked(const __fp16* a, const __fp16* b, float* c, int n) {
  constexpr int BS = 64;
  std::fill(c, c + static_cast<size_t>(n) * static_cast<size_t>(n), 0.0f);

  for (int ii = 0; ii < n; ii += BS) {
    const int i_end = std::min(ii + BS, n);
    for (int kk = 0; kk < n; kk += BS) {
      const int k_end = std::min(kk + BS, n);
      for (int jj = 0; jj < n; jj += BS) {
        const int j_end = std::min(jj + BS, n);

        for (int i = ii; i < i_end; i++) {
          const __fp16* a_row = a + static_cast<size_t>(i) * n;
          float* c_row = c + static_cast<size_t>(i) * n;
          for (int k = kk; k < k_end; k++) {
            const float a_ik = static_cast<float>(a_row[k]);
            const __fp16* b_row = b + static_cast<size_t>(k) * n;
            for (int j = jj; j < j_end; j++) {
              c_row[j] += a_ik * static_cast<float>(b_row[j]);
            }
          }
        }
      }
    }
  }
}

void gemm_bf16_blocked(const uint16_t* a, const uint16_t* b, float* c, int n) {
  constexpr int BS = 64;
  std::fill(c, c + static_cast<size_t>(n) * static_cast<size_t>(n), 0.0f);

  for (int ii = 0; ii < n; ii += BS) {
    const int i_end = std::min(ii + BS, n);
    for (int kk = 0; kk < n; kk += BS) {
      const int k_end = std::min(kk + BS, n);
      for (int jj = 0; jj < n; jj += BS) {
        const int j_end = std::min(jj + BS, n);

        for (int i = ii; i < i_end; i++) {
          const uint16_t* a_row = a + static_cast<size_t>(i) * n;
          float* c_row = c + static_cast<size_t>(i) * n;
          for (int k = kk; k < k_end; k++) {
            const float a_ik = bf16_bits_to_float(a_row[k]);
            const uint16_t* b_row = b + static_cast<size_t>(k) * n;
            for (int j = jj; j < j_end; j++) {
              c_row[j] += a_ik * bf16_bits_to_float(b_row[j]);
            }
          }
        }
      }
    }
  }
}

void gemm_int8_blocked(const int8_t* a, const int8_t* b, int32_t* c, int n) {
  constexpr int BS = 64;
  std::fill(c, c + static_cast<size_t>(n) * static_cast<size_t>(n), 0);

  for (int ii = 0; ii < n; ii += BS) {
    const int i_end = std::min(ii + BS, n);
    for (int kk = 0; kk < n; kk += BS) {
      const int k_end = std::min(kk + BS, n);
      for (int jj = 0; jj < n; jj += BS) {
        const int j_end = std::min(jj + BS, n);

        for (int i = ii; i < i_end; i++) {
          const int8_t* a_row = a + static_cast<size_t>(i) * n;
          int32_t* c_row = c + static_cast<size_t>(i) * n;
          for (int k = kk; k < k_end; k++) {
            const int32_t a_ik = static_cast<int32_t>(a_row[k]);
            const int8_t* b_row = b + static_cast<size_t>(k) * n;
            for (int j = jj; j < j_end; j++) {
              c_row[j] += a_ik * static_cast<int32_t>(b_row[j]);
            }
          }
        }
      }
    }
  }
}

void gemm_int8_ref_plain(const int8_t* a, const int8_t* b, int32_t* c, int n) {
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  std::fill(c, c + nn, 0);
  for (int i = 0; i < n; i++) {
    const int8_t* a_row = a + static_cast<size_t>(i) * n;
    int32_t* c_row = c + static_cast<size_t>(i) * n;
    for (int k = 0; k < n; k++) {
      const int32_t a_ik = static_cast<int32_t>(a_row[k]);
      const int8_t* b_row = b + static_cast<size_t>(k) * n;
      for (int j = 0; j < n; j++) {
        c_row[j] += a_ik * static_cast<int32_t>(b_row[j]);
      }
    }
  }
}

__attribute__((target("fullfp16")))
float dot_fp16_neon(const __fp16* a, const __fp16* b, int n) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  int k = 0;
  for (; k + 8 <= n; k += 8) {
    const auto av = vld1q_f16(reinterpret_cast<const float16_t*>(a + k));
    const auto bv = vld1q_f16(reinterpret_cast<const float16_t*>(b + k));
    acc = vfmlalq_low_f16(acc, av, bv);
    acc = vfmlalq_high_f16(acc, av, bv);
  }
  float sum = vaddvq_f32(acc);
  for (; k < n; k++) {
    sum += static_cast<float>(a[k]) * static_cast<float>(b[k]);
  }
  return sum;
}

__attribute__((target("dotprod")))
int32_t dot_int8_neon(const int8_t* a, const int8_t* b, int n) {
  int32x4_t acc = vdupq_n_s32(0);
  int k = 0;
  for (; k + 16 <= n; k += 16) {
    const int8x16_t av = vld1q_s8(a + k);
    const int8x16_t bv = vld1q_s8(b + k);
    acc = vdotq_s32(acc, av, bv);
  }
  int32_t sum = vaddvq_s32(acc);
  for (; k < n; k++) {
    sum += static_cast<int32_t>(a[k]) * static_cast<int32_t>(b[k]);
  }
  return sum;
}

void gemm_fp16_neon_pretransposed(const __fp16* a, const __fp16* bt, float* c, int n) {
  for (int i = 0; i < n; i++) {
    const __fp16* a_row = a + static_cast<size_t>(i) * n;
    float* c_row = c + static_cast<size_t>(i) * n;
    for (int j = 0; j < n; j++) {
      c_row[j] = dot_fp16_neon(a_row, bt + static_cast<size_t>(j) * n, n);
    }
  }
}

void gemm_int8_neon_pretransposed(const int8_t* a, const int8_t* bt, int32_t* c, int n) {
  for (int i = 0; i < n; i++) {
    const int8_t* a_row = a + static_cast<size_t>(i) * n;
    int32_t* c_row = c + static_cast<size_t>(i) * n;
    for (int j = 0; j < n; j++) {
      c_row[j] = dot_int8_neon(a_row, bt + static_cast<size_t>(j) * n, n);
    }
  }
}

double seconds_since(const std::chrono::high_resolution_clock::time_point& start,
                     const std::chrono::high_resolution_clock::time_point& end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

double tflops_for_gemm(int n, double seconds) {
  // 2*N^3 ops, -> 1e12 ops/s. Output label decides TFLOPS vs TOPS.
  const double nn = static_cast<double>(n);
  const double flops = 2.0 * nn * nn * nn;
  return flops / seconds / 1e12;
}

struct RunResult { double best_seconds = 0.0; double tflops = 0.0; };

template <typename Fn>
RunResult bench_gemm(Fn&& fn, const float* a, const float* b, float* c, int n, int warmup,
                     int repeats) {
  for (int i = 0; i < warmup; i++) fn(a, b, c, n);

  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < repeats; i++) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    fn(a, b, c, n);
    const auto t1 = std::chrono::high_resolution_clock::now();
    best = std::min(best, seconds_since(t0, t1));
  }

  RunResult r;
  r.best_seconds = best;
  r.tflops = tflops_for_gemm(n, best);
  return r;
}

template <typename Fn>
RunResult bench_callable(Fn&& fn, int n, int warmup, int repeats) {
  for (int i = 0; i < warmup; i++) fn();

  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < repeats; i++) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    const auto t1 = std::chrono::high_resolution_clock::now();
    best = std::min(best, seconds_since(t0, t1));
  }

  RunResult r;
  r.best_seconds = best;
  r.tflops = tflops_for_gemm(n, best);
  return r;
}

double max_abs_diff(const float* x, const float* y, int n) {
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  double m = 0.0;
  for (size_t i = 0; i < nn; i++) {
    m = std::max(m, static_cast<double>(std::fabs(x[i] - y[i])));
  }
  return m;
}

int64_t max_abs_diff_i32(const int32_t* x, const int32_t* y, int n) {
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  int64_t m = 0;
  for (size_t i = 0; i < nn; i++) {
    const int64_t d = static_cast<int64_t>(x[i]) - static_cast<int64_t>(y[i]);
    m = std::max<int64_t>(m, std::llabs(d));
  }
  return m;
}

void print_env_hint() {
  // 不强制修改线程策略，但提示可复现性相关变量。
  const char* v1 = std::getenv("VECLIB_MAXIMUM_THREADS");
  const char* v2 = std::getenv("VECLIB_NUM_THREADS");
  if (v1 || v2) {
    std::cout << "env: VECLIB_MAXIMUM_THREADS=" << (v1 ? v1 : "(unset)")
              << " VECLIB_NUM_THREADS=" << (v2 ? v2 : "(unset)") << "\n";
  } else {
    std::cout << "hint: 可设置 VECLIB_MAXIMUM_THREADS 影响 Accelerate 线程数（用于复现）。\n";
  }
}

void print_logo() {
  // 纯文本标题用于避免不同终端字体下的 ASCII Art 误读（例如把 T 看成 F）。
  std::cout << "MTFLOPS\n";
  std::cout << " __  __ _______ ______ _      ____  _____   _____ \n";
  std::cout << "|  \\/  |__   __|  ____| |    / __ \\|  __ \\ / ____|\n";
  std::cout << "| \\  / |  | |  | |__  | |   | |  | | |__) | (___  \n";
  std::cout << "| |\\/| |  | |  |  __| | |   | |  | |  ___/ \\___ \\ \n";
  std::cout << "| |  | |  | |  | |    | |___| |__| | |     ____) |\n";
  std::cout << "|_|  |_|  |_|  |_|    |______\\____/|_|    |_____/ \n\n";
}

std::string precision_to_string(Precision p) {
  switch (p) {
    case Precision::NA: return "-";
    case Precision::FP16: return "FP16";
    case Precision::FP32: return "FP32";
    case Precision::BF16: return "BF16";
    case Precision::INT8: return "INT8";
  }
  return "-";
}

std::string metric_to_string(Precision p) {
  return (p == Precision::INT8) ? "TOPS" : "TFLOPS";
}

std::optional<double> read_power_watts_powermetrics(std::string& note) {
  // powermetrics 通常需要 sudo。这里仅在用户显式启用 `--power powermetrics` 时尝试读取，
  // 失败则返回空并给出 note（避免影响基准测试主流程）。
  if (geteuid() != 0) {
    note = "powermetrics 需要 sudo（用 sudo 运行本程序后再启用 --power powermetrics）";
    return std::nullopt;
  }
  auto run_cmd = [](const char* cmd, std::string& out) -> int {
    FILE* fp = popen(cmd, "r");
    if (!fp) return -1;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), fp)) out += buf;
    const int rc = pclose(fp);
    return rc;
  };

  auto parse_gpu_watts = [](const std::string& text) -> std::optional<double> {
    // 逐行：找包含 gpu 且包含 W/mW 的行，提取第一个数值。
    size_t start = 0;
    while (start < text.size()) {
      size_t end = text.find('\n', start);
      if (end == std::string::npos) end = text.size();
      std::string line = text.substr(start, end - start);
      start = end + 1;

      std::string lower = line;
      for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      if (lower.find("gpu") == std::string::npos) continue;

      const bool has_mw = (lower.find("mw") != std::string::npos);
      const bool has_w = (lower.find('w') != std::string::npos);
      if (!has_w) continue;

      size_t i = 0;
      while (i < line.size() && (line[i] < '0' || line[i] > '9') && line[i] != '.' && line[i] != '-') i++;
      if (i >= line.size()) continue;
      const double v = std::strtod(line.c_str() + i, nullptr);
      if (!std::isfinite(v) || v <= 0.0) continue;

      double watts = v;
      if (has_mw) watts = v / 1000.0;
      if (watts > 0.0 && watts < 200.0) return watts;
    }
    return std::nullopt;
  };

  // 优先尝试 gpu_power sampler（部分系统字段更直接），失败再退回 smc sampler。
  {
    std::string out;
    const int rc = run_cmd("powermetrics --samplers gpu_power -n 1 -i 200 2>&1", out);
    if (rc == 0) {
      if (auto w = parse_gpu_watts(out)) {
        note = "powermetrics(gpu_power)";
        return w;
      }
    }
  }
  {
    std::string out;
    const int rc = run_cmd("powermetrics --samplers smc -n 1 -i 200 2>&1", out);
    if (rc == 0) {
      if (auto w = parse_gpu_watts(out)) {
        note = "powermetrics(smc)";
        return w;
      }
    }
  }

  note = "powermetrics 未解析到 GPU 功耗字段";
  return std::nullopt;
}

void print_table_header(bool show_watts) {
  std::cout << std::left
            << std::setw(8) << "N"
            << std::setw(20) << "Unit"
            << std::setw(8) << "Prec"
            << std::setw(10) << "ms"
            << std::setw(10) << "Score"
            << std::setw(8) << "Metric";
  if (show_watts) {
    std::cout << std::setw(10) << "Watts" << std::setw(12) << "GOPS/W";
  }
  std::cout << "Note\n";
}

void print_row(const BenchRow& r, bool show_watts) {
  const double ms = r.seconds * 1000.0;
  std::cout << std::left
            << std::setw(8) << r.n
            << std::setw(20) << r.unit
            << std::setw(8) << precision_to_string(r.precision)
            << std::setw(10) << std::fixed << std::setprecision(3) << ms
            << std::setw(10) << std::fixed << std::setprecision(3) << r.tflops
            << std::setw(8) << metric_to_string(r.precision);

  if (show_watts) {
    if (r.watts.has_value()) {
      const double gops = r.tflops * 1000.0;
      const double eff = (*r.watts > 0.0) ? (gops / *r.watts) : 0.0;
      std::cout << std::setw(10) << std::fixed << std::setprecision(2) << *r.watts
                << std::setw(12) << std::fixed << std::setprecision(2) << eff;
    } else {
      std::cout << std::setw(10) << "-" << std::setw(12) << "-";
    }
  }

  std::string note = r.note;
  if (r.throttling) {
    if (!note.empty()) note += " | ";
    note += "Thermal Throttling";
  }
  std::cout << note << "\n";
}

std::vector<int> make_sweep_sizes(int min_n, int max_n) {
  std::vector<int> sizes;
  if (min_n > max_n) std::swap(min_n, max_n);
  int n = min_n;
  // KISS：默认走 2x 递增，覆盖 512..16384 的“典型甜点区”。
  while (n <= max_n) {
    sizes.push_back(n);
    if (n > max_n / 2) break;
    n *= 2;
  }
  if (sizes.empty()) {
    sizes.push_back(max_n);
  } else if (sizes.back() != max_n) {
    // 尽量包含 max_n（若是 2^k 序列外）。
    sizes.push_back(max_n);
  }
  // 去重
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  return sizes;
}

struct SweepKey {
  std::string unit;
  Precision precision = Precision::NA;
};

std::string sweep_key_to_string(const SweepKey& k) {
  return k.unit + " " + precision_to_string(k.precision);
}

struct SweetSpot {
  int n = 0;
  double tflops = 0.0;
  double peak_tflops = 0.0;
};

std::optional<SweetSpot> compute_sweet_spot(const std::vector<BenchRow>& rows, const SweepKey& key) {
  std::vector<const BenchRow*> candidates;
  candidates.reserve(rows.size());
  double peak = 0.0;
  for (const auto& r : rows) {
    if (r.unit != key.unit) continue;
    if (r.precision != key.precision) continue;
    peak = std::max(peak, r.tflops);
    candidates.push_back(&r);
  }
  if (candidates.empty() || peak <= 0.0) return std::nullopt;

  // 规则（KISS）：选“最小的 N”，其 score ≥ 98% 峰值，且单次耗时 ≥ 2ms（避免计时过短噪声）。
  const double threshold = peak * 0.98;
  const double min_seconds = 0.002;

  const BenchRow* best = nullptr;
  for (const auto* r : candidates) {
    if (r->tflops < threshold) continue;
    if (r->seconds < min_seconds) continue;
    if (!best || r->n < best->n) best = r;
  }
  if (!best) {
    // 兜底：直接选峰值对应的 N（若多个，取最小 N）
    for (const auto* r : candidates) {
      if (r->tflops == peak) {
        if (!best || r->n < best->n) best = r;
      }
    }
  }
  if (!best) return std::nullopt;

  SweetSpot s;
  s.n = best->n;
  s.tflops = best->tflops;
  s.peak_tflops = peak;
  return s;
}

bool verify_cpu_precision(Precision p, int n, double& diff, int64_t& diff_i32) {
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  diff = 0.0;
  diff_i32 = 0;

  if (p == Precision::FP32) {
    float* a = aligned_alloc_64<float>(nn);
    float* b = aligned_alloc_64<float>(nn);
    float* c1 = aligned_alloc_64<float>(nn);
    float* c2 = aligned_alloc_64<float>(nn);
    fill_matrix(a, n, 123);
    fill_matrix(b, n, 456);
    gemm_amx_accelerate(a, b, c1, n);
    gemm_ref_blocked(a, b, c2, n);
    diff = max_abs_diff(c1, c2, n);
    std::free(a);
    std::free(b);
    std::free(c1);
    std::free(c2);
    return diff <= 1e-3;
  }

  if (p == Precision::FP16) {
    __fp16* a16 = aligned_alloc_64<__fp16>(nn);
    __fp16* b16 = aligned_alloc_64<__fp16>(nn);
    __fp16* bt16 = aligned_alloc_64<__fp16>(nn);
    float* c1 = aligned_alloc_64<float>(nn);
    float* c2 = aligned_alloc_64<float>(nn);
    float* a = aligned_alloc_64<float>(nn);
    float* b = aligned_alloc_64<float>(nn);
    fill_matrix_fp16(a16, n, 123);
    fill_matrix_fp16(b16, n, 456);
    transpose_square(b16, bt16, n);
    for (size_t i = 0; i < nn; i++) {
      a[i] = static_cast<float>(a16[i]);
      b[i] = static_cast<float>(b16[i]);
    }
    if (cpu_has_fp16()) gemm_fp16_neon_pretransposed(a16, bt16, c1, n);
    else gemm_fp16_blocked(a16, b16, c1, n);
    gemm_ref_blocked(a, b, c2, n);
    diff = max_abs_diff(c1, c2, n);
    std::free(a16);
    std::free(b16);
    std::free(bt16);
    std::free(c1);
    std::free(c2);
    std::free(a);
    std::free(b);
    return diff <= 1e-2;
  }

  if (p == Precision::BF16) {
    uint16_t* a16 = aligned_alloc_64<uint16_t>(nn);
    uint16_t* b16 = aligned_alloc_64<uint16_t>(nn);
    float* c1 = aligned_alloc_64<float>(nn);
    float* c2 = aligned_alloc_64<float>(nn);
    float* a = aligned_alloc_64<float>(nn);
    float* b = aligned_alloc_64<float>(nn);
    fill_matrix_bf16(a16, n, 123);
    fill_matrix_bf16(b16, n, 456);
    for (size_t i = 0; i < nn; i++) {
      a[i] = bf16_bits_to_float(a16[i]);
      b[i] = bf16_bits_to_float(b16[i]);
    }
    gemm_bf16_blocked(a16, b16, c1, n);
    gemm_ref_blocked(a, b, c2, n);
    diff = max_abs_diff(c1, c2, n);
    std::free(a16);
    std::free(b16);
    std::free(c1);
    std::free(c2);
    std::free(a);
    std::free(b);
    return diff <= 1e-2;
  }

  int8_t* a = aligned_alloc_64<int8_t>(nn);
  int8_t* b = aligned_alloc_64<int8_t>(nn);
  int8_t* bt = aligned_alloc_64<int8_t>(nn);
  int32_t* c1 = aligned_alloc_64<int32_t>(nn);
  int32_t* c2 = aligned_alloc_64<int32_t>(nn);
  fill_matrix_int8(a, n, 123);
  fill_matrix_int8(b, n, 456);
  transpose_square(b, bt, n);
  if (cpu_has_dotprod()) gemm_int8_neon_pretransposed(a, bt, c1, n);
  else gemm_int8_blocked(a, b, c1, n);
  gemm_int8_ref_plain(a, b, c2, n);
  diff_i32 = max_abs_diff_i32(c1, c2, n);
  std::free(a);
  std::free(b);
  std::free(bt);
  std::free(c1);
  std::free(c2);
  return diff_i32 == 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);

  print_logo();
  if (opt.unit == Options::Unit::GPU && opt.gpu_workload == Options::GpuWorkload::Peak) {
    std::cout << "Peak score: FP/BF16 use matrix-unit FLOPs; INT8 uses a scalar TOPS probe.\n\n";
  } else {
    std::cout << "GEMM score: 2*N^3 operations (FP/BF16 => TFLOPS, INT8 => TOPS)\n\n";
  }
  std::cout << std::flush;

  const int n = opt.n;
  std::array<bool, 4> gpu_verified = {false, false, false, false};

  auto get_watts = [&](std::string& note) -> std::optional<double> {
    if (opt.power == Options::Power::None) return std::nullopt;
    if (opt.power == Options::Power::Powermetrics) return read_power_watts_powermetrics(note);
    return std::nullopt;
  };

  auto run_one_gpu = [&](int size, GpuPrecision precision, BenchRow& row, std::string& err) -> bool {
    GpuBenchOptions gopt;
    gopt.n = size;
    gopt.warmup = opt.warmup;
    gopt.repeats = opt.repeats;
    gopt.precision = precision;
    gopt.shader_path = opt.gpu_shader_path;
    gopt.batch = opt.gpu_batch;
    gopt.inner = opt.gpu_inner;
    gopt.storage = (opt.gpu_storage == Options::GpuStorage::Private) ? GpuStorageMode::Private
                                                                     : GpuStorageMode::Shared;
    gopt.workload = (opt.gpu_workload == Options::GpuWorkload::Peak) ? GpuWorkload::Peak
                                                                     : GpuWorkload::Gemm;
    if (opt.gpu_kernel == Options::GpuKernel::V4) gopt.kernel = GpuKernelVariant::V4;
    else gopt.kernel = GpuKernelVariant::Auto;

    const Precision row_precision = precision_from_gpu(precision);
    const int pidx = precision_index(row_precision);
    if (opt.verify && !gpu_verified[pidx] && gopt.workload == GpuWorkload::Gemm) {
      // KISS：仅校验一次小规模正确性，避免在 sweep/stress 中重复校验影响性能。
      const int vn = 128;
      std::vector<float> gpu_c;

      GpuBenchOptions vopt = gopt;
      vopt.n = vn;
      vopt.warmup = 0;
      vopt.repeats = 1;
      vopt.batch = 1;
      vopt.inner = 1;
      vopt.storage = GpuStorageMode::Shared;  // 读回需要 shared
      vopt.workload = GpuWorkload::Gemm;
      vopt.readback_c = &gpu_c;

      GpuBenchResult vout;
      std::string verr;
      if (!run_gpu_bench(vopt, vout, verr)) {
        err = "GPU verify failed: " + verr;
        return false;
      }

      std::vector<float> ref_c(static_cast<size_t>(vn) * static_cast<size_t>(vn));
      if (precision == GpuPrecision::FP32) {
        std::vector<float> a(static_cast<size_t>(vn) * static_cast<size_t>(vn));
        std::vector<float> b(static_cast<size_t>(vn) * static_cast<size_t>(vn));
        fill_matrix(a.data(), vn, 1);
        fill_matrix(b.data(), vn, 2);
        gemm_ref_blocked(a.data(), b.data(), ref_c.data(), vn);
      } else if (precision == GpuPrecision::FP16) {
        std::vector<__fp16> a16(static_cast<size_t>(vn) * static_cast<size_t>(vn));
        std::vector<__fp16> b16(static_cast<size_t>(vn) * static_cast<size_t>(vn));
        fill_matrix_fp16(a16.data(), vn, 1);
        fill_matrix_fp16(b16.data(), vn, 2);
        std::vector<float> a(static_cast<size_t>(vn) * static_cast<size_t>(vn));
        std::vector<float> b(static_cast<size_t>(vn) * static_cast<size_t>(vn));
        for (size_t i = 0; i < a.size(); i++) {
          a[i] = static_cast<float>(a16[i]);
          b[i] = static_cast<float>(b16[i]);
        }
        gemm_ref_blocked(a.data(), b.data(), ref_c.data(), vn);
      } else if (precision == GpuPrecision::BF16) {
        std::vector<float> a(static_cast<size_t>(vn) * static_cast<size_t>(vn));
        std::vector<float> b(static_cast<size_t>(vn) * static_cast<size_t>(vn));
        fill_matrix_bf16_as_float(a.data(), vn, 1);
        fill_matrix_bf16_as_float(b.data(), vn, 2);
        gemm_ref_blocked(a.data(), b.data(), ref_c.data(), vn);
      } else {
        std::vector<int8_t> a(static_cast<size_t>(vn) * static_cast<size_t>(vn));
        std::vector<int8_t> b(static_cast<size_t>(vn) * static_cast<size_t>(vn));
        std::vector<int32_t> c(static_cast<size_t>(vn) * static_cast<size_t>(vn));
        fill_matrix_int8(a.data(), vn, 1);
        fill_matrix_int8(b.data(), vn, 2);
        gemm_int8_blocked(a.data(), b.data(), c.data(), vn);
        for (size_t i = 0; i < c.size(); i++) ref_c[i] = static_cast<float>(c[i]);
      }

      const double diff = max_abs_diff(ref_c.data(), gpu_c.data(), vn);
      const double thr = (precision == GpuPrecision::FP32) ? 1e-2
                         : (precision == GpuPrecision::INT8) ? 0.0
                                                            : 1e-1;
      std::cout << "verify(gpu): prec=" << precision_to_string(row_precision)
                << " N=" << vn << " max_abs_diff=" << std::scientific << diff << "\n\n";
      if (diff > thr) {
        err = "GPU verify max_abs_diff too high";
        return false;
      }
      gpu_verified[pidx] = true;
    }

    GpuBenchResult gout;
    if (!run_gpu_bench(gopt, gout, err)) return false;

    row.n = size;
    row.unit = (precision == GpuPrecision::INT8) ? "GPU (Metal)" : "GPU (simdgroup)";
    row.precision = row_precision;
    row.seconds = gout.best_seconds;
    row.tflops = gout.score;
    std::string k;
    if (gout.used_kernel == GpuKernelVariant::V4) k = "kernel=v4";
    if (!k.empty()) {
      if (!row.note.empty()) row.note += " | ";
      row.note += k;
    }
    if (opt.gpu_batch > 1) {
      if (!row.note.empty()) row.note += " | ";
      row.note += "batch=" + std::to_string(opt.gpu_batch);
    }
    if (opt.gpu_inner > 1) {
      if (!row.note.empty()) row.note += " | ";
      row.note += "inner=" + std::to_string(opt.gpu_inner);
    }
    if (opt.gpu_workload == Options::GpuWorkload::Peak) {
      if (!row.note.empty()) row.note += " | ";
      row.note += "workload=peak";
    }
    return true;
  };

  auto run_one_cpu = [&](int size, Precision precision, Options::Mode mode, BenchRow& row) -> void {
    const size_t nn = static_cast<size_t>(size) * static_cast<size_t>(size);
    auto add_note = [&](std::string_view s) {
      if (!row.note.empty()) row.note += " | ";
      row.note += std::string(s);
    };

    int repeats = opt.repeats;
    if ((mode == Options::Mode::Ref || precision != Precision::FP32) && size >= 1536 &&
        opt.repeats > 1 && !opt.sweep) {
      repeats = 1;
    }

    RunResult r;
    if (precision == Precision::FP32) {
      float* a = aligned_alloc_64<float>(nn);
      float* b = aligned_alloc_64<float>(nn);
      float* c = aligned_alloc_64<float>(nn);
      fill_matrix(a, size, 1);
      fill_matrix(b, size, 2);
      if (mode == Options::Mode::AMX) {
        r = bench_gemm(gemm_amx_accelerate, a, b, c, size, opt.warmup, repeats);
        row.unit = "AMX (cblas)";
      } else {
        r = bench_gemm(gemm_ref_blocked, a, b, c, size, opt.warmup, repeats);
        row.unit = "CPU ref";
      }
      std::free(a);
      std::free(b);
      std::free(c);
    } else if (precision == Precision::FP16) {
      __fp16* a = aligned_alloc_64<__fp16>(nn);
      __fp16* b = aligned_alloc_64<__fp16>(nn);
      __fp16* bt = aligned_alloc_64<__fp16>(nn);
      float* c = aligned_alloc_64<float>(nn);
      fill_matrix_fp16(a, size, 1);
      fill_matrix_fp16(b, size, 2);
      transpose_square(b, bt, size);
      const bool fast = cpu_has_fp16();
      if (fast) {
        r = bench_callable([&]() { gemm_fp16_neon_pretransposed(a, bt, c, size); }, size, opt.warmup,
                           repeats);
      } else {
        r = bench_callable([&]() { gemm_fp16_blocked(a, b, c, size); }, size, opt.warmup, repeats);
      }
      row.unit = fast ? "CPU NEON" : "CPU kernel";
      add_note("fp32 accumulate");
      if (!fast) add_note("fallback=blocked");
      std::free(a);
      std::free(b);
      std::free(bt);
      std::free(c);
    } else if (precision == Precision::BF16) {
      uint16_t* a = aligned_alloc_64<uint16_t>(nn);
      uint16_t* b = aligned_alloc_64<uint16_t>(nn);
      float* c = aligned_alloc_64<float>(nn);
      fill_matrix_bf16(a, size, 1);
      fill_matrix_bf16(b, size, 2);
      const bool fast = cpu_use_bf16_neon();
      r = bench_callable([&]() { gemm_bf16_blocked(a, b, c, size); }, size, opt.warmup, repeats);
      row.unit = fast ? "CPU NEON" : "CPU kernel";
      add_note("fp32 accumulate");
      if (!fast) add_note("fallback=blocked");
      std::free(a);
      std::free(b);
      std::free(c);
    } else {
      int8_t* a = aligned_alloc_64<int8_t>(nn);
      int8_t* b = aligned_alloc_64<int8_t>(nn);
      int8_t* bt = aligned_alloc_64<int8_t>(nn);
      int32_t* c = aligned_alloc_64<int32_t>(nn);
      fill_matrix_int8(a, size, 1);
      fill_matrix_int8(b, size, 2);
      transpose_square(b, bt, size);
      const bool fast = cpu_has_dotprod();
      if (fast) {
        r = bench_callable([&]() { gemm_int8_neon_pretransposed(a, bt, c, size); }, size, opt.warmup,
                           repeats);
      } else {
        r = bench_callable([&]() { gemm_int8_blocked(a, b, c, size); }, size, opt.warmup, repeats);
      }
      row.unit = fast ? "CPU NEON" : "CPU kernel";
      add_note("int32 accumulate");
      if (!fast) add_note("fallback=blocked");
      std::free(a);
      std::free(b);
      std::free(bt);
      std::free(c);
    }

    row.n = size;
    row.precision = precision;
    row.seconds = r.best_seconds;
    row.tflops = r.tflops;
  };

  if (opt.sweep) {
    const auto sizes = make_sweep_sizes(opt.sweep_min, opt.sweep_max);
    const bool show_watts = (opt.power != Options::Power::None);
    print_table_header(show_watts);

    std::vector<BenchRow> printed;
    const size_t per_size = opt.precisions.size() * 2u;
    printed.reserve(static_cast<size_t>(sizes.size()) * per_size);

    for (int size : sizes) {
      std::string note;
      const std::optional<double> watts = get_watts(note);

      if (opt.unit == Options::Unit::GPU) {
        bool stop = false;
        for (Precision p : opt.precisions) {
          BenchRow row;
          row.n = size;
          row.watts = watts;
          row.note = note;
          std::string err;
          if (!run_one_gpu(size, gpu_precision_from_precision(p), row, err)) {
            row.note = err;
            print_row(row, show_watts);
            printed.push_back(row);
            stop = true;
            break;
          }
          print_row(row, show_watts);
          printed.push_back(row);
        }
        if (stop) break;
      } else {
        std::cout << std::flush;
        const bool run_amx = (opt.mode == Options::Mode::AMX || opt.mode == Options::Mode::Both);
        const bool run_ref = (opt.mode == Options::Mode::Ref || opt.mode == Options::Mode::Both);

        for (Precision p : opt.precisions) {
          BenchRow base;
          base.n = size;
          base.watts = watts;
          base.note = note;
          if (opt.verify) {
            const int vn = std::min(size, 128);
            double diff = 0.0;
            int64_t diff_i32 = 0;
            if (!verify_cpu_precision(p, vn, diff, diff_i32)) {
              if (!base.note.empty()) base.note += " | ";
              base.note += (p == Precision::INT8) ? ("verify max_abs_diff_i32=" + std::to_string(diff_i32))
                                                  : "verify max_abs_diff too high";
            }
          }

          if (p == Precision::FP32) {
            if (run_amx) {
              BenchRow r1 = base;
              run_one_cpu(size, p, Options::Mode::AMX, r1);
              print_row(r1, show_watts);
              printed.push_back(r1);
            }
            if (run_ref) {
              BenchRow r2 = base;
              run_one_cpu(size, p, Options::Mode::Ref, r2);
              print_row(r2, show_watts);
              printed.push_back(r2);
            }
          } else {
            BenchRow r = base;
            run_one_cpu(size, p, Options::Mode::AMX, r);
            print_row(r, show_watts);
            printed.push_back(r);
          }
        }
        continue;
      }
    }

    // sweet spot 推荐（同一 unit+precision 下）
    std::vector<SweepKey> keys;
    if (opt.unit == Options::Unit::GPU) {
      for (Precision p : opt.precisions) {
        keys.push_back({(p == Precision::INT8) ? "GPU (Metal)" : "GPU (simdgroup)", p});
      }
    } else {
      for (Precision p : opt.precisions) {
        if (p == Precision::FP32) {
          if (opt.mode == Options::Mode::AMX || opt.mode == Options::Mode::Both) {
            keys.push_back({"AMX (cblas)", p});
          }
          if (opt.mode == Options::Mode::Ref || opt.mode == Options::Mode::Both) {
            keys.push_back({"CPU ref", p});
          }
        } else {
          const bool fast = (p == Precision::FP16) ? cpu_has_fp16()
                            : (p == Precision::BF16) ? cpu_use_bf16_neon()
                                                     : cpu_has_dotprod();
          keys.push_back({fast ? "CPU NEON" : "CPU kernel", p});
        }
      }
    }

    bool printed_any = false;
    for (const auto& k : keys) {
      if (auto ss = compute_sweet_spot(printed, k)) {
        if (!printed_any) {
          std::cout << "\nrecommend:\n";
          printed_any = true;
        }
        const double pct = (ss->peak_tflops > 0.0) ? (ss->tflops / ss->peak_tflops * 100.0) : 0.0;
        std::cout << "- " << sweep_key_to_string(k) << " sweet spot N=" << ss->n
                  << " (" << std::fixed << std::setprecision(3) << ss->tflops
                  << " " << metric_to_string(k.precision) << ", " << std::setprecision(1)
                  << pct << "% of peak)\n";
      }
    }
    return 0;
  }

  if (opt.stress > 0) {
    const bool show_watts = (opt.power != Options::Power::None);
    print_table_header(show_watts);

    std::optional<double> cached_watts;
    std::string cached_power_note;

    if (opt.unit == Options::Unit::GPU) {
      std::array<double, 4> baseline = {0.0, 0.0, 0.0, 0.0};
      std::array<int, 4> throttle_streak = {0, 0, 0, 0};

      for (int iter = 0; iter < opt.stress; iter++) {
        std::string note;
        if (show_watts) {
          if (iter % opt.power_every == 0 || !cached_watts.has_value()) {
            cached_watts = get_watts(note);
            cached_power_note = note;
          }
        }

        bool stop = false;
        for (Precision p : opt.precisions) {
          BenchRow row;
          row.n = n;
          row.note = "iter=" + std::to_string(iter);
          if (show_watts) {
            row.watts = cached_watts;
            if (!cached_power_note.empty()) row.note += " | " + cached_power_note;
          }

          std::string err;
          if (!run_one_gpu(n, gpu_precision_from_precision(p), row, err)) {
            row.note = err;
            print_row(row, show_watts);
            stop = true;
            break;
          }

          const int idx = precision_index(p);
          baseline[idx] = std::max(baseline[idx], row.tflops);
          if (baseline[idx] > 0.0 && row.tflops < baseline[idx] * 0.9) throttle_streak[idx]++;
          else throttle_streak[idx] = 0;
          row.throttling = (throttle_streak[idx] >= 2);
          print_row(row, show_watts);
        }
        if (stop) break;
      }
      return 0;
    }

    // CPU stress：每个 precision 跑一个主路径；FP32 优先 AMX，除非用户只选 ref。
    std::array<double, 4> baseline = {0.0, 0.0, 0.0, 0.0};
    std::array<int, 4> throttle_streak = {0, 0, 0, 0};
    for (int iter = 0; iter < opt.stress; iter++) {
      std::string note;
      if (show_watts) {
        if (iter % opt.power_every == 0 || !cached_watts.has_value()) {
          cached_watts = get_watts(note);
          cached_power_note = note;
        }
      }

      const bool run_amx = (opt.mode == Options::Mode::AMX || opt.mode == Options::Mode::Both);
      for (Precision p : opt.precisions) {
        BenchRow row;
        row.n = n;
        row.note = "iter=" + std::to_string(iter);
        if (show_watts) {
          row.watts = cached_watts;
          if (!cached_power_note.empty()) row.note += " | " + cached_power_note;
        }

        const auto which = (p == Precision::FP32 && !run_amx) ? Options::Mode::Ref : Options::Mode::AMX;
        run_one_cpu(n, p, which, row);

        const int idx = precision_index(p);
        baseline[idx] = std::max(baseline[idx], row.tflops);
        if (baseline[idx] > 0.0 && row.tflops < baseline[idx] * 0.9) throttle_streak[idx]++;
        else throttle_streak[idx] = 0;
        row.throttling = (throttle_streak[idx] >= 2);
        print_row(row, show_watts);
      }
    }
    return 0;
  }

  if (opt.unit == Options::Unit::GPU) {
    print_table_header(false);
    for (Precision p : opt.precisions) {
      BenchRow row;
      std::string err;
      if (!run_one_gpu(n, gpu_precision_from_precision(p), row, err)) {
        std::cerr << "GPU benchmark failed: " << err << "\n";
        return 2;
      }
      print_row(row, false);
    }
    return 0;
  }

  std::cout << "CPU (FP32 Accelerate cblas + NEON/typed kernels)\n";
  print_env_hint();
  std::cout << "\n";

  const bool run_amx = (opt.mode == Options::Mode::AMX || opt.mode == Options::Mode::Both);
  const bool run_ref = (opt.mode == Options::Mode::Ref || opt.mode == Options::Mode::Both);

  if (opt.verify) {
    const int vn = std::min(n, 128);
    for (Precision p : opt.precisions) {
      double diff = 0.0;
      int64_t diff_i32 = 0;
      const bool ok = verify_cpu_precision(p, vn, diff, diff_i32);
      std::cout << "verify(cpu): prec=" << precision_to_string(p) << " N=" << vn;
      if (p == Precision::INT8) {
        std::cout << " max_abs_diff_i32=" << diff_i32;
      } else {
        std::cout << " max_abs_diff=" << std::scientific << diff;
      }
      std::cout << (ok ? "\n" : " FAILED\n");
    }
    std::cout << "\n";
  }

  print_table_header(false);
  for (Precision p : opt.precisions) {
    if (p == Precision::FP32) {
      if (run_amx) {
        BenchRow row;
        run_one_cpu(n, p, Options::Mode::AMX, row);
        print_row(row, false);
      }
      if (run_ref) {
        BenchRow row;
        run_one_cpu(n, p, Options::Mode::Ref, row);
        print_row(row, false);
      }
    } else {
      BenchRow row;
      run_one_cpu(n, p, Options::Mode::AMX, row);
      print_row(row, false);
    }
  }

  return 0;
}
