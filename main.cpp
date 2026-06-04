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
#include <future>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <pthread.h>
#include <string>
#include <string_view>
#include <sys/sysctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <arm_neon.h>
#include <arm_sme.h>
#include <arm_sve.h>

#include "gpu_bench.h"
#include "npu_bench.h"

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
  enum class Unit { CPU, GPU, NPU };
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
  std::string precision_arg;
  bool n_set = false;

  std::vector<Precision> precisions;
  std::string gpu_shader_path = "shaders/gemm.metal";
  GpuKernel gpu_kernel = GpuKernel::Auto;
  GpuStorage gpu_storage = GpuStorage::Private;
  int gpu_batch = 1;
  int gpu_inner = 1;
  GpuWorkload gpu_workload = GpuWorkload::Gemm;
  int npu_spatial = 64;
  int npu_depth = 128;

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

NpuPrecision npu_precision_from_precision(Precision p) {
  if (p == Precision::FP16) return NpuPrecision::FP16;
  if (p == Precision::FP32) die("NPU FP32 score is disabled: ANE does not expose native FP32 compute");
  if (p == Precision::BF16) return NpuPrecision::BF16;
  if (p == Precision::INT8) return NpuPrecision::INT8;
  die("invalid NPU precision");
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
          << "MTFLOPS (CPU SME + Metal GPU + private ANE)\n\n"
          << "Usage:\n"
          << "  ./mtflops [--unit cpu|gpu|npu] [--mode amx|ref|both] [--precision fp16|fp32|bf16|int8|both|all]\n"
          << "           [--n N] [--warmup W] [--repeats R] [--verify 0|1]\n"
          << "           [--shader <path>]\n\n"
          << "           [--sweep 0|1] [--sweep-min N] [--sweep-max N]\n"
          << "           [--stress ITERS]\n"
          << "           [--power none|powermetrics] [--power-every K]\n"
          << "           [--kernel auto|v4]\n\n"
          << "           [--gpu-storage shared|private] [--gpu-batch B]\n\n"
          << "           [--gpu-inner I]  (I>1 为 compute-amplified，趋近理论峰值用)\n\n"
          << "           [--gpu-workload gemm|peak]\n\n"
          << "           [--npu-spatial S] [--npu-depth D]\n\n"
          << "Notes:\n"
          << "  - CPU 口径：SME MOPA peak probe；N 控制 inner loop 规模，不再走 Accelerate/NEON/I8MM GEMM。\n"
          << "  - GPU GEMM 口径：浮点按 2*N^3 计 TFLOPS；INT8 按 2*N^3 计 TOPS。\n"
          << "  - NPU 使用 private ANE conv1x1-chain：只报告 FP16/INT8；N=channels，--npu-spatial/--npu-depth 控制工作量。\n";
      std::exit(0);
    } else if (a == "--unit") {
      auto v = need_value("--unit");
      if (v == "cpu") opt.unit = Options::Unit::CPU;
      else if (v == "gpu") opt.unit = Options::Unit::GPU;
      else if (v == "npu" || v == "ane") opt.unit = Options::Unit::NPU;
      else die("--unit must be one of: cpu, gpu, npu");
    } else if (a == "--mode") {
      auto v = need_value("--mode");
      if (v == "amx") opt.mode = Options::Mode::AMX;
      else if (v == "ref") opt.mode = Options::Mode::Ref;
      else if (v == "both") opt.mode = Options::Mode::Both;
      else die("--mode must be one of: amx, ref, both");
    } else if (a == "--precision") {
      auto v = need_value("--precision");
      opt.precision_arg = std::string(v);
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
    } else if (a == "--npu-spatial") {
      int v = 0;
      if (!parse_int(need_value("--npu-spatial"), v) || v <= 0) die("--npu-spatial must be > 0");
      opt.npu_spatial = v;
    } else if (a == "--npu-depth") {
      int v = 0;
      if (!parse_int(need_value("--npu-depth"), v) || v <= 0) die("--npu-depth must be > 0");
      opt.npu_depth = v;
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
      opt.n_set = true;
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
    if (opt.unit == Options::Unit::GPU) opt.precisions = {Precision::FP16};
    else if (opt.unit == Options::Unit::NPU) opt.precisions = {Precision::FP16, Precision::INT8};
    else opt.precisions = {Precision::FP32};
  } else if (opt.unit == Options::Unit::NPU) {
    if (opt.precision_arg == "all" || opt.precision_arg == "both") {
      opt.precisions = {Precision::FP16, Precision::INT8};
    } else {
      for (Precision p : opt.precisions) {
        if (p == Precision::FP32) {
          die("NPU FP32 score is disabled: true FP32 ANE conv is not accepted; use fp16 or int8");
        }
      }
    }
  }
  if (opt.unit == Options::Unit::NPU && !opt.n_set) opt.n = 512;
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

bool cpu_has_i8mm() {
  static const bool yes = cpu_feature_enabled("hw.optional.arm.FEAT_I8MM");
  return yes;
}

[[maybe_unused]] bool cpu_use_bf16_neon() {
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

void gemm_fp16_neon_pretransposed(const __fp16* a, const __fp16* bt, float* c, int n) {
  for (int i = 0; i < n; i++) {
    const __fp16* a_row = a + static_cast<size_t>(i) * n;
    float* c_row = c + static_cast<size_t>(i) * n;
    for (int j = 0; j < n; j++) {
      c_row[j] = dot_fp16_neon(a_row, bt + static_cast<size_t>(j) * n, n);
    }
  }
}

size_t i8mm_bpack_size(int n) {
  const size_t j_tiles = static_cast<size_t>(n / 4);
  const size_t k_blocks = static_cast<size_t>(n / 8);
  return j_tiles * k_blocks * 32u;
}

void pack_b_int8_i8mm_4col(const int8_t* b, int8_t* bpack, int n) {
  const int j_tiles = n / 4;
  const int k_blocks = n / 8;
  for (int jt = 0; jt < j_tiles; jt++) {
    const int j0 = jt * 4;
    for (int kb = 0; kb < k_blocks; kb++) {
      const int k0 = kb * 8;
      int8_t* dst = bpack + (static_cast<size_t>(jt) * k_blocks + kb) * 32u;
      for (int kk = 0; kk < 8; kk++) {
        const int8_t* brow = b + static_cast<size_t>(k0 + kk) * n + j0;
        dst[kk] = brow[0];
        dst[8 + kk] = brow[1];
        dst[16 + kk] = brow[2];
        dst[24 + kk] = brow[3];
      }
    }
  }
}

inline void store_i8mm_2x2(int32x4_t acc, int32_t* c, int n, int row, int col) {
  c[static_cast<size_t>(row) * n + col] = vgetq_lane_s32(acc, 0);
  c[static_cast<size_t>(row) * n + col + 1] = vgetq_lane_s32(acc, 1);
  c[static_cast<size_t>(row + 1) * n + col] = vgetq_lane_s32(acc, 2);
  c[static_cast<size_t>(row + 1) * n + col + 1] = vgetq_lane_s32(acc, 3);
}

__attribute__((target("i8mm")))
void gemm_int8_i8mm_4x4_kernel(const int8_t* a, const int8_t* bpack, int32_t* c, int n,
                               int k_blocks) {
  int32x4_t c00 = vdupq_n_s32(0);
  int32x4_t c02 = vdupq_n_s32(0);
  int32x4_t c20 = vdupq_n_s32(0);
  int32x4_t c22 = vdupq_n_s32(0);

  for (int kb = 0; kb < k_blocks; kb++) {
    const int k0 = kb * 8;
    const int8x8_t a0 = vld1_s8(a + 0 * n + k0);
    const int8x8_t a1 = vld1_s8(a + 1 * n + k0);
    const int8x8_t a2 = vld1_s8(a + 2 * n + k0);
    const int8x8_t a3 = vld1_s8(a + 3 * n + k0);
    const int8x16_t a01 = vcombine_s8(a0, a1);
    const int8x16_t a23 = vcombine_s8(a2, a3);
    const int8_t* bp = bpack + static_cast<size_t>(kb) * 32u;
    const int8x16_t b01 = vld1q_s8(bp);
    const int8x16_t b23 = vld1q_s8(bp + 16);

    c00 = vmmlaq_s32(c00, a01, b01);
    c02 = vmmlaq_s32(c02, a01, b23);
    c20 = vmmlaq_s32(c20, a23, b01);
    c22 = vmmlaq_s32(c22, a23, b23);
  }

  store_i8mm_2x2(c00, c, n, 0, 0);
  store_i8mm_2x2(c02, c, n, 0, 2);
  store_i8mm_2x2(c20, c, n, 2, 0);
  store_i8mm_2x2(c22, c, n, 2, 2);
}

void add_int8_tail_k_4x4(const int8_t* a, const int8_t* b, int32_t* c, int n, int k0, int j0) {
  for (int r = 0; r < 4; r++) {
    for (int col = 0; col < 4; col++) {
      int32_t sum = c[static_cast<size_t>(r) * n + col];
      for (int k = k0; k < n; k++) {
        sum += static_cast<int32_t>(a[static_cast<size_t>(r) * n + k]) *
               static_cast<int32_t>(b[static_cast<size_t>(k) * n + j0 + col]);
      }
      c[static_cast<size_t>(r) * n + col] = sum;
    }
  }
}

void gemm_int8_i8mm_prepacked_row_tiles(const int8_t* a, const int8_t* b, const int8_t* bpack,
                                        int32_t* c, int n, int tile_begin, int tile_end) {
  const int j_tiles = n / 4;
  const int k_blocks = n / 8;
  const int k_tail = k_blocks * 8;
  const int j_main = j_tiles * 4;

  for (int it = tile_begin; it < tile_end; it++) {
    const int i = it * 4;
    const int8_t* a_tile = a + static_cast<size_t>(i) * n;
    int32_t* c_tile = c + static_cast<size_t>(i) * n;
    for (int jt = 0; jt < j_tiles; jt++) {
      const int j = jt * 4;
      int32_t* c_block = c_tile + j;
      if (k_blocks > 0) {
        const int8_t* bp = bpack + static_cast<size_t>(jt) * k_blocks * 32u;
        gemm_int8_i8mm_4x4_kernel(a_tile, bp, c_block, n, k_blocks);
      } else {
        for (int r = 0; r < 4; r++) {
          for (int col = 0; col < 4; col++) c_block[static_cast<size_t>(r) * n + col] = 0;
        }
      }
      if (k_tail < n) add_int8_tail_k_4x4(a_tile, b, c_block, n, k_tail, j);
    }

    for (int r = 0; r < 4; r++) {
      const int8_t* a_row = a + static_cast<size_t>(i + r) * n;
      int32_t* c_row = c + static_cast<size_t>(i + r) * n;
      for (int j = j_main; j < n; j++) {
        int32_t sum = 0;
        for (int k = 0; k < n; k++) {
          sum += static_cast<int32_t>(a_row[k]) * static_cast<int32_t>(b[static_cast<size_t>(k) * n + j]);
        }
        c_row[j] = sum;
      }
    }
  }
}

void gemm_int8_scalar_tail_rows(const int8_t* a, const int8_t* b, int32_t* c, int n, int i_begin) {
  for (int i = i_begin; i < n; i++) {
    const int8_t* a_row = a + static_cast<size_t>(i) * n;
    int32_t* c_row = c + static_cast<size_t>(i) * n;
    for (int j = 0; j < n; j++) {
      int32_t sum = 0;
      for (int k = 0; k < n; k++) {
        sum += static_cast<int32_t>(a_row[k]) * static_cast<int32_t>(b[static_cast<size_t>(k) * n + j]);
      }
      c_row[j] = sum;
    }
  }
}

int gemm_int8_i8mm_tiled_prepacked_threaded(const int8_t* a, const int8_t* b, const int8_t* bpack,
                                            int32_t* c, int n) {
  const int row_tiles = n / 4;
  const int tail_row_begin = row_tiles * 4;
  const unsigned hw_threads = std::thread::hardware_concurrency();
  const int threads = std::max(1, std::min(row_tiles, static_cast<int>(std::max(1u, hw_threads))));

  if (row_tiles > 0) {
    if (threads == 1) {
      gemm_int8_i8mm_prepacked_row_tiles(a, b, bpack, c, n, 0, row_tiles);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; t++) {
        const int begin = row_tiles * t / threads;
        const int end = row_tiles * (t + 1) / threads;
        workers.emplace_back([=]() {
          gemm_int8_i8mm_prepacked_row_tiles(a, b, bpack, c, n, begin, end);
        });
      }
      for (auto& w : workers) w.join();
    }
  }

  if (tail_row_begin < n) gemm_int8_scalar_tail_rows(a, b, c, n, tail_row_begin);
  return threads;
}

void gemm_int8_i8mm_tiled(const int8_t* a, const int8_t* b, int32_t* c, int n) {
  std::vector<int8_t> bpack(i8mm_bpack_size(n));
  if (!bpack.empty()) pack_b_int8_i8mm_4col(b, bpack.data(), n);
  gemm_int8_i8mm_tiled_prepacked_threaded(a, b, bpack.data(), c, n);
}

double seconds_since(const std::chrono::high_resolution_clock::time_point& start,
                     const std::chrono::high_resolution_clock::time_point& end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

[[maybe_unused]] double tflops_for_gemm(int n, double seconds) {
  // 2*N^3 ops, -> 1e12 ops/s. Output label decides TFLOPS vs TOPS.
  const double nn = static_cast<double>(n);
  const double flops = 2.0 * nn * nn * nn;
  return flops / seconds / 1e12;
}

[[maybe_unused]] uint64_t ops_for_gemm_u64(int n) {
  const uint64_t nn = static_cast<uint64_t>(n);
  return 2ull * nn * nn * nn;
}

struct RunResult {
  double best_seconds = 0.0;
  double tflops = 0.0;
  uint64_t ops = 0;
  int threads = 1;
};

constexpr size_t kSmePeakScratchElems = 4096;
constexpr size_t kSmePeakTileElems = 65536;
constexpr uint64_t kSmePeakIterScale = 16384;
constexpr uint64_t kSmePeakMaxInner = 1ull << 28;

int cpu_sme_thread_count() {
  if (const char* env = std::getenv("MTFLOPS_CPU_THREADS")) {
    int v = 0;
    if (parse_int(env, v) && v > 0) return v;
  }
  const unsigned hc = std::thread::hardware_concurrency();
  return static_cast<int>(std::max(1u, hc) * 6u);
}

void set_cpu_sme_worker_qos() {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

uint64_t cpu_sme_peak_inner_from_n(int n) {
  const uint64_t nn = static_cast<uint64_t>(std::max(1, n));
  return std::min(nn * kSmePeakIterScale, kSmePeakMaxInner);
}

uint16_t fp32_to_bf16_peak_bits(float f) {
  return float_to_bf16_bits(f);
}

// These kernels are intentionally peak probes, not full GEMM kernels. They run the
// public SME outer-product instructions directly with stable in-register inputs.
__arm_new("za") __arm_locally_streaming __attribute__((target("sme,bf16,i8mm"), noinline))
void cpu_sme_peak_fp32_kernel(const float* a, const float* b, float* out, uint64_t inner,
                              uint64_t* ops_per_mopa, uint32_t* lanes32) {
  const svbool_t pg = svptrue_b32();
  const svfloat32_t va = svld1_f32(pg, a);
  const svfloat32_t vb = svld1_f32(pg, b);
  const uint64_t n32 = svcntsw();
  *lanes32 = static_cast<uint32_t>(n32);
  *ops_per_mopa = 2ull * n32 * n32;
  svzero_za();
  for (uint64_t i = 0; i < inner; i++) {
    svmopa_za32_m(0, pg, pg, va, vb);
  }
  for (uint32_t r = 0; r < n32; r++) {
    svst1_hor_za32(0, r, pg, out + static_cast<size_t>(r) * n32);
  }
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme,bf16,i8mm"), noinline))
void cpu_sme_peak_fp16_kernel(const __fp16* a, const __fp16* b, float* out, uint64_t inner,
                              uint64_t* ops_per_mopa, uint32_t* lanes32) {
  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg32 = svptrue_b32();
  const svfloat16_t va = svld1_f16(pg16, a);
  const svfloat16_t vb = svld1_f16(pg16, b);
  const uint64_t n32 = svcntsw();
  *lanes32 = static_cast<uint32_t>(n32);
  *ops_per_mopa = 4ull * n32 * n32;  // 2 FP16 K lanes per FP32 tile cell, GEMM counts mul+add.
  svzero_za();
  for (uint64_t i = 0; i < inner; i++) {
    svmopa_za32_m(0, pg16, pg16, va, vb);
  }
  for (uint32_t r = 0; r < n32; r++) {
    svst1_hor_za32(0, r, pg32, out + static_cast<size_t>(r) * n32);
  }
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme,bf16,i8mm"), noinline))
void cpu_sme_peak_bf16_kernel(const bfloat16_t* a, const bfloat16_t* b, float* out, uint64_t inner,
                              uint64_t* ops_per_mopa, uint32_t* lanes32) {
  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg32 = svptrue_b32();
  const svbfloat16_t va = svld1_bf16(pg16, a);
  const svbfloat16_t vb = svld1_bf16(pg16, b);
  const uint64_t n32 = svcntsw();
  *lanes32 = static_cast<uint32_t>(n32);
  *ops_per_mopa = 4ull * n32 * n32;  // 2 BF16 K lanes per FP32 tile cell, GEMM counts mul+add.
  svzero_za();
  for (uint64_t i = 0; i < inner; i++) {
    svmopa_za32_m(0, pg16, pg16, va, vb);
  }
  for (uint32_t r = 0; r < n32; r++) {
    svst1_hor_za32(0, r, pg32, out + static_cast<size_t>(r) * n32);
  }
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme,bf16,i8mm"), noinline))
void cpu_sme_peak_int8_kernel(const int8_t* a, const int8_t* b, int32_t* out, uint64_t inner,
                              uint64_t* ops_per_mopa, uint32_t* lanes32) {
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svint8_t va = svld1_s8(pg8, a);
  const svint8_t vb = svld1_s8(pg8, b);
  const uint64_t n32 = svcntsw();
  *lanes32 = static_cast<uint32_t>(n32);
  *ops_per_mopa = 8ull * n32 * n32;  // 4 INT8 K lanes per INT32 tile cell, TOPS counts mul+add.
  svzero_za();
  for (uint64_t i = 0; i < inner; i++) {
    svmopa_za32_m(0, pg8, pg8, va, vb);
  }
  for (uint32_t r = 0; r < n32; r++) {
    svst1_hor_za32(0, r, pg32, out + static_cast<size_t>(r) * n32);
  }
}

struct CpuSmeVerifyResult {
  bool ok = false;
  double diff = 0.0;
  int64_t diff_i32 = 0;
};

RunResult bench_cpu_sme_peak(Precision p, int n, int warmup, int repeats, std::string& note) {
  const uint64_t inner = cpu_sme_peak_inner_from_n(n);
  const int threads = cpu_sme_thread_count();
  uint64_t ops_per_mopa = 0;
  uint32_t lanes32 = 0;

  RunResult r;
  auto finish = [&](double best_seconds) {
    r.best_seconds = best_seconds;
    r.ops = ops_per_mopa * inner * static_cast<uint64_t>(threads);
    r.tflops = static_cast<double>(r.ops) / best_seconds / 1e12;
    r.threads = threads;
    note = "sme_mopa peak_probe";
    note += " | inner=" + std::to_string(inner);
    note += " | threads=" + std::to_string(threads);
    note += " | svl=" + std::to_string(static_cast<unsigned>(lanes32 * sizeof(float))) + "B";
    if (p == Precision::INT8) note += " | int32 accumulate";
    else note += " | fp32 accumulate";
  };

  if (p == Precision::FP32) {
    float* a = aligned_alloc_64<float>(kSmePeakScratchElems);
    float* b = aligned_alloc_64<float>(kSmePeakScratchElems);
    std::vector<float*> outs(static_cast<size_t>(threads));
    for (size_t i = 0; i < kSmePeakScratchElems; i++) {
      a[i] = 1.0f;
      b[i] = 1.0f;
    }
    for (float*& out : outs) out = aligned_alloc_64<float>(kSmePeakTileElems);
    auto run_parallel = [&]() {
      std::vector<std::thread> workers;
      std::vector<uint64_t> ops(static_cast<size_t>(threads), 0);
      std::vector<uint32_t> lanes(static_cast<size_t>(threads), 0);
      workers.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; t++) {
        workers.emplace_back([&, t]() {
          set_cpu_sme_worker_qos();
          cpu_sme_peak_fp32_kernel(a, b, outs[t], inner, &ops[t], &lanes[t]);
        });
      }
      for (auto& worker : workers) worker.join();
      ops_per_mopa = ops[0];
      lanes32 = lanes[0];
    };
    for (int i = 0; i < warmup; i++) run_parallel();
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeats; i++) {
      const auto t0 = std::chrono::high_resolution_clock::now();
      run_parallel();
      const auto t1 = std::chrono::high_resolution_clock::now();
      best = std::min(best, seconds_since(t0, t1));
    }
    volatile float sink = outs[0][0];
    (void)sink;
    std::free(a);
    std::free(b);
    for (float* out : outs) std::free(out);
    finish(best);
    return r;
  }

  if (p == Precision::FP16) {
    __fp16* a = aligned_alloc_64<__fp16>(kSmePeakScratchElems);
    __fp16* b = aligned_alloc_64<__fp16>(kSmePeakScratchElems);
    std::vector<float*> outs(static_cast<size_t>(threads));
    for (size_t i = 0; i < kSmePeakScratchElems; i++) {
      a[i] = static_cast<__fp16>(1.0f);
      b[i] = static_cast<__fp16>(1.0f);
    }
    for (float*& out : outs) out = aligned_alloc_64<float>(kSmePeakTileElems);
    auto run_parallel = [&]() {
      std::vector<std::thread> workers;
      std::vector<uint64_t> ops(static_cast<size_t>(threads), 0);
      std::vector<uint32_t> lanes(static_cast<size_t>(threads), 0);
      workers.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; t++) {
        workers.emplace_back([&, t]() {
          set_cpu_sme_worker_qos();
          cpu_sme_peak_fp16_kernel(a, b, outs[t], inner, &ops[t], &lanes[t]);
        });
      }
      for (auto& worker : workers) worker.join();
      ops_per_mopa = ops[0];
      lanes32 = lanes[0];
    };
    for (int i = 0; i < warmup; i++) run_parallel();
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeats; i++) {
      const auto t0 = std::chrono::high_resolution_clock::now();
      run_parallel();
      const auto t1 = std::chrono::high_resolution_clock::now();
      best = std::min(best, seconds_since(t0, t1));
    }
    volatile float sink = outs[0][0];
    (void)sink;
    std::free(a);
    std::free(b);
    for (float* out : outs) std::free(out);
    finish(best);
    return r;
  }

  if (p == Precision::BF16) {
    bfloat16_t* a = aligned_alloc_64<bfloat16_t>(kSmePeakScratchElems);
    bfloat16_t* b = aligned_alloc_64<bfloat16_t>(kSmePeakScratchElems);
    std::vector<float*> outs(static_cast<size_t>(threads));
    const uint16_t one_bits = fp32_to_bf16_peak_bits(1.0f);
    for (size_t i = 0; i < kSmePeakScratchElems; i++) {
      std::memcpy(&a[i], &one_bits, sizeof(one_bits));
      std::memcpy(&b[i], &one_bits, sizeof(one_bits));
    }
    for (float*& out : outs) out = aligned_alloc_64<float>(kSmePeakTileElems);
    auto run_parallel = [&]() {
      std::vector<std::thread> workers;
      std::vector<uint64_t> ops(static_cast<size_t>(threads), 0);
      std::vector<uint32_t> lanes(static_cast<size_t>(threads), 0);
      workers.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; t++) {
        workers.emplace_back([&, t]() {
          set_cpu_sme_worker_qos();
          cpu_sme_peak_bf16_kernel(a, b, outs[t], inner, &ops[t], &lanes[t]);
        });
      }
      for (auto& worker : workers) worker.join();
      ops_per_mopa = ops[0];
      lanes32 = lanes[0];
    };
    for (int i = 0; i < warmup; i++) run_parallel();
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < repeats; i++) {
      const auto t0 = std::chrono::high_resolution_clock::now();
      run_parallel();
      const auto t1 = std::chrono::high_resolution_clock::now();
      best = std::min(best, seconds_since(t0, t1));
    }
    volatile float sink = outs[0][0];
    (void)sink;
    std::free(a);
    std::free(b);
    for (float* out : outs) std::free(out);
    finish(best);
    return r;
  }

  int8_t* a = aligned_alloc_64<int8_t>(kSmePeakScratchElems);
  int8_t* b = aligned_alloc_64<int8_t>(kSmePeakScratchElems);
  std::vector<int32_t*> outs(static_cast<size_t>(threads));
  for (size_t i = 0; i < kSmePeakScratchElems; i++) {
    a[i] = 1;
    b[i] = 1;
  }
  for (int32_t*& out : outs) out = aligned_alloc_64<int32_t>(kSmePeakTileElems);
  auto run_parallel = [&]() {
    std::vector<std::thread> workers;
    std::vector<uint64_t> ops(static_cast<size_t>(threads), 0);
    std::vector<uint32_t> lanes(static_cast<size_t>(threads), 0);
    workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; t++) {
      workers.emplace_back([&, t]() {
        set_cpu_sme_worker_qos();
        cpu_sme_peak_int8_kernel(a, b, outs[t], inner, &ops[t], &lanes[t]);
      });
    }
    for (auto& worker : workers) worker.join();
    ops_per_mopa = ops[0];
    lanes32 = lanes[0];
  };
  for (int i = 0; i < warmup; i++) run_parallel();
  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < repeats; i++) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    run_parallel();
    const auto t1 = std::chrono::high_resolution_clock::now();
    best = std::min(best, seconds_since(t0, t1));
  }
  volatile int32_t sink = outs[0][0];
  (void)sink;
  std::free(a);
  std::free(b);
  for (int32_t* out : outs) std::free(out);
  finish(best);
  return r;
}

CpuSmeVerifyResult verify_cpu_sme_peak(Precision p) {
  constexpr uint64_t inner = 8;
  uint64_t ops_per_mopa = 0;
  uint32_t lanes32 = 0;
  CpuSmeVerifyResult v;

  if (p == Precision::FP32) {
    float* a = aligned_alloc_64<float>(kSmePeakScratchElems);
    float* b = aligned_alloc_64<float>(kSmePeakScratchElems);
    float* out = aligned_alloc_64<float>(kSmePeakTileElems);
    for (size_t i = 0; i < kSmePeakScratchElems; i++) a[i] = b[i] = 1.0f;
    std::memset(out, 0, kSmePeakTileElems * sizeof(float));
    cpu_sme_peak_fp32_kernel(a, b, out, inner, &ops_per_mopa, &lanes32);
    const double expected = static_cast<double>(inner);
    v.diff = std::fabs(static_cast<double>(out[0]) - expected);
    v.ok = v.diff <= 1e-5;
    std::free(a);
    std::free(b);
    std::free(out);
    return v;
  }

  if (p == Precision::FP16) {
    __fp16* a = aligned_alloc_64<__fp16>(kSmePeakScratchElems);
    __fp16* b = aligned_alloc_64<__fp16>(kSmePeakScratchElems);
    float* out = aligned_alloc_64<float>(kSmePeakTileElems);
    for (size_t i = 0; i < kSmePeakScratchElems; i++) {
      a[i] = static_cast<__fp16>(1.0f);
      b[i] = static_cast<__fp16>(1.0f);
    }
    std::memset(out, 0, kSmePeakTileElems * sizeof(float));
    cpu_sme_peak_fp16_kernel(a, b, out, inner, &ops_per_mopa, &lanes32);
    const double expected = static_cast<double>(inner * 2);
    v.diff = std::fabs(static_cast<double>(out[0]) - expected);
    v.ok = v.diff <= 1e-5;
    std::free(a);
    std::free(b);
    std::free(out);
    return v;
  }

  if (p == Precision::BF16) {
    bfloat16_t* a = aligned_alloc_64<bfloat16_t>(kSmePeakScratchElems);
    bfloat16_t* b = aligned_alloc_64<bfloat16_t>(kSmePeakScratchElems);
    float* out = aligned_alloc_64<float>(kSmePeakTileElems);
    const uint16_t one_bits = fp32_to_bf16_peak_bits(1.0f);
    for (size_t i = 0; i < kSmePeakScratchElems; i++) {
      std::memcpy(&a[i], &one_bits, sizeof(one_bits));
      std::memcpy(&b[i], &one_bits, sizeof(one_bits));
    }
    std::memset(out, 0, kSmePeakTileElems * sizeof(float));
    cpu_sme_peak_bf16_kernel(a, b, out, inner, &ops_per_mopa, &lanes32);
    const double expected = static_cast<double>(inner * 2);
    v.diff = std::fabs(static_cast<double>(out[0]) - expected);
    v.ok = v.diff <= 1e-5;
    std::free(a);
    std::free(b);
    std::free(out);
    return v;
  }

  int8_t* a = aligned_alloc_64<int8_t>(kSmePeakScratchElems);
  int8_t* b = aligned_alloc_64<int8_t>(kSmePeakScratchElems);
  int32_t* out = aligned_alloc_64<int32_t>(kSmePeakTileElems);
  for (size_t i = 0; i < kSmePeakScratchElems; i++) a[i] = b[i] = 1;
  std::memset(out, 0, kSmePeakTileElems * sizeof(int32_t));
  cpu_sme_peak_int8_kernel(a, b, out, inner, &ops_per_mopa, &lanes32);
  const int64_t expected = static_cast<int64_t>(inner * 4);
  v.diff_i32 = std::llabs(static_cast<int64_t>(out[0]) - expected);
  v.ok = v.diff_i32 == 0;
  std::free(a);
  std::free(b);
  std::free(out);
  return v;
}

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
  r.ops = ops_for_gemm_u64(n);
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
  r.ops = ops_for_gemm_u64(n);
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

[[maybe_unused]] void print_env_hint() {
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

const char* power_key_for_unit(Options::Unit unit) {
  switch (unit) {
    case Options::Unit::CPU: return "cpu_power";
    case Options::Unit::GPU: return "gpu_power";
    case Options::Unit::NPU: return "ane_power";
  }
  return "cpu_power";
}

const char* power_unit_name(Options::Unit unit) {
  switch (unit) {
    case Options::Unit::CPU: return "cpu";
    case Options::Unit::GPU: return "gpu";
    case Options::Unit::NPU: return "ane";
  }
  return "cpu";
}

std::optional<double> parse_plist_power_watts(const std::string& text, const char* key) {
  const std::string marker = std::string("<key>") + key + "</key>";
  std::optional<double> last;
  size_t pos = 0;
  while ((pos = text.find(marker, pos)) != std::string::npos) {
    const size_t value_start = pos + marker.size();
    const size_t next_key = text.find("<key>", value_start);
    const size_t real_pos = text.find("<real>", value_start);
    const size_t int_pos = text.find("<integer>", value_start);

    size_t tag_pos = std::string::npos;
    const char* open_tag = nullptr;
    const char* close_tag = nullptr;
    if (real_pos != std::string::npos && (next_key == std::string::npos || real_pos < next_key)) {
      tag_pos = real_pos;
      open_tag = "<real>";
      close_tag = "</real>";
    }
    if (int_pos != std::string::npos && (next_key == std::string::npos || int_pos < next_key) &&
        (tag_pos == std::string::npos || int_pos < tag_pos)) {
      tag_pos = int_pos;
      open_tag = "<integer>";
      close_tag = "</integer>";
    }
    if (!open_tag) {
      pos = value_start;
      continue;
    }

    const size_t number_start = tag_pos + std::strlen(open_tag);
    const size_t number_end = text.find(close_tag, number_start);
    if (number_end == std::string::npos) {
      pos = number_start;
      continue;
    }
    const std::string number = text.substr(number_start, number_end - number_start);
    char* endp = nullptr;
    const double mw = std::strtod(number.c_str(), &endp);
    if (endp != number.c_str() && std::isfinite(mw) && mw >= 0.0 && mw < 200000.0) {
      last = mw / 1000.0;
    }
    pos = number_end + std::strlen(close_tag);
  }
  return last;
}

void append_power_rail_note(std::string& note, const std::string& sample) {
  const auto cpu = parse_plist_power_watts(sample, "cpu_power");
  const auto gpu = parse_plist_power_watts(sample, "gpu_power");
  const auto ane = parse_plist_power_watts(sample, "ane_power");
  char buf[160];
  std::snprintf(buf, sizeof(buf), "cpu=%.3fW gpu=%.3fW ane=%.3fW",
                cpu.value_or(0.0), gpu.value_or(0.0), ane.value_or(0.0));
  note += " | ";
  note += buf;
}

int run_command_capture(const std::string& cmd, std::string& out) {
  FILE* fp = popen(cmd.c_str(), "r");
  if (!fp) return -1;
  char buf[4096];
  while (std::fgets(buf, sizeof(buf), fp)) out += buf;
  return pclose(fp);
}

std::optional<double> read_power_watts_powermetrics(Options::Unit unit, std::string& note) {
  // powermetrics 通常需要 sudo。这里仅在用户显式启用 `--power powermetrics` 时尝试读取，
  // 失败则返回空并给出 note（避免影响基准测试主流程）。
  if (geteuid() != 0) {
    note = "powermetrics 需要 sudo（用 sudo 运行本程序后再启用 --power powermetrics）";
    return std::nullopt;
  }

  auto parse_text_watts = [](const std::string& text, const char* needle) -> std::optional<double> {
    // 逐行：找包含目标 rail 名称且包含 W/mW 的行，提取第一个数值。
    size_t start = 0;
    while (start < text.size()) {
      size_t end = text.find('\n', start);
      if (end == std::string::npos) end = text.size();
      std::string line = text.substr(start, end - start);
      start = end + 1;

      std::string lower = line;
      for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      if (lower.find(needle) == std::string::npos) continue;

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

  const char* key = power_key_for_unit(unit);
  const char* name = power_unit_name(unit);
  const int interval_ms = (unit == Options::Unit::NPU) ? 1000 : 200;

  // cpu_power 的 plist 在本机同时包含 cpu_power/gpu_power/ane_power，且单位为 mW。
  // 这是 NPU/ANE rail 当前可解析的最小 sampler。
  {
    std::string out;
    const std::string cmd =
        "powermetrics --samplers cpu_power -f plist -n 1 -i " + std::to_string(interval_ms) + " 2>&1";
    const int rc = run_command_capture(cmd, out);
    if (rc == 0) {
      if (auto w = parse_plist_power_watts(out, key)) {
        note = "powermetrics(cpu_power plist key=" + std::string(key) + " interval=" +
               std::to_string(interval_ms) + "ms)";
        append_power_rail_note(note, out);
        return w;
      }
    }
  }

  // GPU 仍保留文本 gpu_power 回退；ANE 的 ane_power 文本 sampler 在本机只输出采样头。
  if (unit == Options::Unit::GPU) {
    std::string out;
    const std::string cmd =
        "powermetrics --samplers gpu_power -n 1 -i " + std::to_string(interval_ms) + " 2>&1";
    const int rc = run_command_capture(cmd, out);
    if (rc == 0) {
      if (auto w = parse_text_watts(out, name)) {
        note = "powermetrics(gpu_power text)";
        return w;
      }
    }
  }

  {
    std::string out;
    const std::string cmd =
        "powermetrics --samplers smc -n 1 -i " + std::to_string(interval_ms) + " 2>&1";
    const int rc = run_command_capture(cmd, out);
    if (rc == 0) {
      if (auto w = parse_text_watts(out, name)) {
        note = "powermetrics(smc text)";
        return w;
      }
    }
  }

  note = "powermetrics 未解析到 " + std::string(key) + " 功耗字段";
  return std::nullopt;
}

std::optional<double> parse_process_gpu_ms_s(const std::string& text, const char* process_name) {
  size_t start = 0;
  while (start < text.size()) {
    size_t end = text.find('\n', start);
    if (end == std::string::npos) end = text.size();
    const std::string line = text.substr(start, end - start);
    start = end + 1;

    size_t i = 0;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) i++;
    const size_t name_len = std::strlen(process_name);
    if (line.compare(i, name_len, process_name) != 0) continue;
    if (i + name_len < line.size() &&
        !std::isspace(static_cast<unsigned char>(line[i + name_len]))) {
      continue;
    }

    bool any = false;
    double last = 0.0;
    const char* p = line.c_str() + i + name_len;
    while (*p) {
      while (*p && !std::isdigit(static_cast<unsigned char>(*p)) && *p != '-' && *p != '+'
             && *p != '.') {
        p++;
      }
      if (!*p) break;
      char* endp = nullptr;
      const double v = std::strtod(p, &endp);
      if (endp != p && std::isfinite(v)) {
        last = v;
        any = true;
        p = endp;
      } else {
        p++;
      }
    }
    if (any) return last;
  }
  return std::nullopt;
}

std::optional<double> read_process_gpu_ms_powermetrics(std::string& note) {
  if (geteuid() != 0) {
    note = "process_gpu 需要 sudo";
    return std::nullopt;
  }

  std::string out;
  const int rc = run_command_capture(
      "powermetrics --samplers tasks --show-process-gpu -n 1 -i 1000 2>&1", out);
  if (rc == 0) {
    if (auto gpu = parse_process_gpu_ms_s(out, "mtflops")) {
      char buf[96];
      std::snprintf(buf, sizeof(buf), "powermetrics(tasks) mtflops_gpu_ms_s=%.2f", *gpu);
      note = buf;
      return gpu;
    }
  }
  note = "powermetrics 未解析到 mtflops GPU ms/s";
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

[[maybe_unused]] bool verify_cpu_precision(Precision p, int n, double& diff, int64_t& diff_i32) {
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
  int32_t* c1 = aligned_alloc_64<int32_t>(nn);
  int32_t* c2 = aligned_alloc_64<int32_t>(nn);
  fill_matrix_int8(a, n, 123);
  fill_matrix_int8(b, n, 456);
  if (cpu_has_i8mm()) gemm_int8_i8mm_tiled(a, b, c1, n);
  else gemm_int8_blocked(a, b, c1, n);
  gemm_int8_ref_plain(a, b, c2, n);
  diff_i32 = max_abs_diff_i32(c1, c2, n);
  std::free(a);
  std::free(b);
  std::free(c1);
  std::free(c2);
  return diff_i32 == 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);

  print_logo();
  if (opt.unit == Options::Unit::CPU) {
    std::cout << "CPU SME peak score: FP/BF16 use SME MOPA FLOPs; INT8 uses SME MOPA TOPS.\n\n";
  } else if (opt.unit == Options::Unit::GPU && opt.gpu_workload == Options::GpuWorkload::Peak) {
    std::cout << "Peak score: FP/BF16 use matrix-unit FLOPs; INT8 uses a scalar TOPS probe.\n\n";
  } else if (opt.unit == Options::Unit::NPU) {
    std::cout << "NPU score: private ANE conv1x1-chain operations (FP16 => TFLOPS, INT8 => TOPS; FP32 disabled)\n\n";
  } else {
    std::cout << "GEMM score: 2*N^3 operations (FP/BF16 => TFLOPS, INT8 => TOPS)\n\n";
  }
  std::cout << std::flush;

  const int n = opt.n;
  std::array<bool, 4> gpu_verified = {false, false, false, false};

  auto get_watts = [&](std::string& note) -> std::optional<double> {
    if (opt.power == Options::Power::None) return std::nullopt;
    if (opt.power == Options::Power::Powermetrics) return read_power_watts_powermetrics(opt.unit, note);
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
    const bool mps_int8_gemm = (precision == GpuPrecision::INT8 && gopt.workload == GpuWorkload::Gemm);
    if (mps_int8_gemm) {
      gopt.batch = 1;
      gopt.inner = 1;
    }
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
    if (precision == GpuPrecision::INT8) {
      row.unit = (gout.backend == "mps-qmm-i32") ? "GPU (MPS QMM)" : "GPU (Metal scalar)";
    } else {
      row.unit = "GPU (simdgroup)";
    }
    row.precision = row_precision;
    row.seconds = gout.best_seconds;
    row.tflops = gout.score;
    std::string k;
    if (gout.used_kernel == GpuKernelVariant::V4) k = "kernel=v4";
    if (!k.empty()) {
      if (!row.note.empty()) row.note += " | ";
      row.note += k;
    }
    if (!gout.backend.empty()) {
      if (!row.note.empty()) row.note += " | ";
      row.note += "backend=" + gout.backend;
    }
    if (mps_int8_gemm && (opt.gpu_batch > 1 || opt.gpu_inner > 1)) {
      if (!row.note.empty()) row.note += " | ";
      row.note += "batch/inner ignored";
    } else if (opt.gpu_batch > 1) {
      if (!row.note.empty()) row.note += " | ";
      row.note += "batch=" + std::to_string(opt.gpu_batch);
    }
    if (!mps_int8_gemm && opt.gpu_inner > 1) {
      if (!row.note.empty()) row.note += " | ";
      row.note += "inner=" + std::to_string(opt.gpu_inner);
    }
    if (opt.gpu_workload == Options::GpuWorkload::Peak) {
      if (!row.note.empty()) row.note += " | ";
      row.note += "workload=peak";
    }
    return true;
  };

  auto run_one_npu = [&](int channels, Precision precision, BenchRow& row, std::string& err) -> bool {
    NpuBenchOptions nopt;
    nopt.channels = channels;
    nopt.spatial = opt.npu_spatial;
    nopt.depth = opt.npu_depth;
    nopt.warmup = opt.warmup;
    nopt.repeats = opt.repeats;
    nopt.precision = npu_precision_from_precision(precision);

    NpuBenchResult nout;
    if (!run_npu_bench(nopt, nout, err)) return false;

    row.n = channels;
    row.unit = "NPU (private ANE)";
    row.precision = precision;
    row.seconds = nout.best_seconds;
    row.tflops = nout.score;
    if (!nout.backend.empty()) {
      if (!row.note.empty()) row.note += " | ";
      row.note += "backend=" + nout.backend;
    }
    if (!nout.note.empty()) {
      if (!row.note.empty()) row.note += " | ";
      row.note += nout.note;
    }
    if (nout.npu_only) {
      if (!row.note.empty()) row.note += " | ";
      row.note += "npu_only=private_ane_eval";
    }
    row.note += " | spatial=" + std::to_string(opt.npu_spatial);
    row.note += " | depth=" + std::to_string(opt.npu_depth);
    return true;
  };

  auto run_one_cpu = [&](int size, Precision precision, Options::Mode /*mode*/, BenchRow& row) -> void {
    auto add_note = [&](std::string_view s) {
      if (!row.note.empty()) row.note += " | ";
      row.note += std::string(s);
    };

    std::string sme_note;
    RunResult r = bench_cpu_sme_peak(precision, size, opt.warmup, opt.repeats, sme_note);
    row.unit = "CPU SME";
    add_note(sme_note);

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
      } else if (opt.unit == Options::Unit::NPU) {
        for (Precision p : opt.precisions) {
          BenchRow row;
          row.n = size;
          row.watts = watts;
          row.note = note;
          std::string err;
          if (!run_one_npu(size, p, row, err)) {
            row.precision = p;
            row.unit = "NPU (private ANE)";
            row.note = err;
          }
          print_row(row, show_watts);
          printed.push_back(row);
        }
      } else {
        std::cout << std::flush;

        for (Precision p : opt.precisions) {
          BenchRow base;
          base.n = size;
          base.watts = watts;
          base.note = note;
          if (opt.verify) {
            const CpuSmeVerifyResult verify = verify_cpu_sme_peak(p);
            if (!verify.ok) {
              if (!base.note.empty()) base.note += " | ";
              base.note += (p == Precision::INT8)
                               ? ("verify_sme diff_i32=" + std::to_string(verify.diff_i32))
                               : ("verify_sme diff=" + std::to_string(verify.diff));
            }
          }

          BenchRow r = base;
          run_one_cpu(size, p, Options::Mode::AMX, r);
          print_row(r, show_watts);
          printed.push_back(r);
        }
        continue;
      }
    }

    // sweet spot 推荐（同一 unit+precision 下）
    std::vector<SweepKey> keys;
    if (opt.unit == Options::Unit::GPU) {
      for (Precision p : opt.precisions) {
        keys.push_back({(p == Precision::INT8) ? "GPU (MPS QMM)" : "GPU (simdgroup)", p});
      }
    } else if (opt.unit == Options::Unit::NPU) {
      for (Precision p : opt.precisions) {
        keys.push_back({"NPU (private ANE)", p});
      }
    } else {
      for (Precision p : opt.precisions) {
        keys.push_back({"CPU SME", p});
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
    std::string cached_process_gpu_note;

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

    if (opt.unit == Options::Unit::NPU) {
      std::array<double, 4> baseline = {0.0, 0.0, 0.0, 0.0};
      std::array<int, 4> throttle_streak = {0, 0, 0, 0};
      auto sample_power_async = [&]() {
        return std::async(std::launch::async, [&]() {
          std::string note;
          auto watts = get_watts(note);
          return std::make_pair(watts, note);
        });
      };
      auto sample_process_gpu_async = [&]() {
        return std::async(std::launch::async, [&]() {
          std::string note;
          auto gpu_ms = read_process_gpu_ms_powermetrics(note);
          return std::make_pair(gpu_ms, note);
        });
      };

      for (int iter = 0; iter < opt.stress; iter++) {
        for (Precision p : opt.precisions) {
          BenchRow row;
          row.n = n;
          row.note = "iter=" + std::to_string(iter);

          const bool sample_power =
              show_watts && (iter % opt.power_every == 0 || !cached_watts.has_value());
          std::future<std::pair<std::optional<double>, std::string>> power_future;
          std::future<std::pair<std::optional<double>, std::string>> process_gpu_future;
          if (sample_power) {
            power_future = sample_power_async();
            process_gpu_future = sample_process_gpu_async();
          }

          std::string err;
          const bool ok = run_one_npu(n, p, row, err);

          if (sample_power) {
            auto sampled = power_future.get();
            cached_watts = sampled.first;
            cached_power_note = sampled.second;
            auto process_gpu = process_gpu_future.get();
            cached_process_gpu_note = process_gpu.second;
          }
          if (show_watts) {
            row.watts = cached_watts;
            if (!cached_power_note.empty()) {
              if (!row.note.empty()) row.note += " | ";
              row.note += cached_power_note;
            }
            if (!cached_process_gpu_note.empty()) {
              if (!row.note.empty()) row.note += " | ";
              row.note += cached_process_gpu_note;
            }
          }

          if (!ok) {
            row.precision = p;
            row.unit = "NPU (private ANE)";
            row.note = err;
            if (!cached_power_note.empty()) row.note += " | " + cached_power_note;
            if (!cached_process_gpu_note.empty()) row.note += " | " + cached_process_gpu_note;
            print_row(row, show_watts);
            continue;
          }

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

  if (opt.unit == Options::Unit::NPU) {
    std::cout << "NPU/ANE private benchmark (conv1x1-chain; N=channels)\n";
    print_table_header(false);
    bool any_fail = false;
    for (Precision p : opt.precisions) {
      BenchRow row;
      std::string err;
      if (!run_one_npu(n, p, row, err)) {
        row.n = n;
        row.unit = "NPU (private ANE)";
        row.precision = p;
        row.note = err;
        any_fail = true;
      }
      print_row(row, false);
    }
    return any_fail ? 2 : 0;
  }

  std::cout << "CPU (SME MOPA peak probe: FP32/FP16/BF16/INT8)\n";
  std::cout << "hint: CPU --n controls SME inner loop scale; MTFLOPS_CPU_THREADS overrides worker count.\n";
  std::cout << "\n";

  if (opt.verify) {
    for (Precision p : opt.precisions) {
      const CpuSmeVerifyResult verify = verify_cpu_sme_peak(p);
      std::cout << "verify(cpu_sme_peak): prec=" << precision_to_string(p);
      if (p == Precision::INT8) {
        std::cout << " diff_i32=" << verify.diff_i32;
      } else {
        std::cout << " diff=" << std::scientific << verify.diff;
      }
      std::cout << (verify.ok ? "\n" : " FAILED\n");
    }
    std::cout << "\n";
  }

  print_table_header(false);
  for (Precision p : opt.precisions) {
    BenchRow row;
    run_one_cpu(n, p, Options::Mode::AMX, row);
    print_row(row, false);
  }

  return 0;
}
