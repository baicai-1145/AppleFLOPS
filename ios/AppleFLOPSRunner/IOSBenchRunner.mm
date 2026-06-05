#import "IOSBenchRunner.h"

#import <Accelerate/Accelerate.h>
#import <UIKit/UIKit.h>
#import <mach/mach_time.h>
#import <setjmp.h>
#import <signal.h>
#import <sys/sysctl.h>

#include <algorithm>
#include <arm_neon.h>
#include <arm_sme.h>
#include <arm_sve.h>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

#include "../../gpu_bench.h"
#include "../../npu_bench.h"

namespace {

sigjmp_buf g_sme_probe_jmp;
volatile sig_atomic_t g_sme_probe_signal = 0;

enum class CpuPrecision { FP32, FP16, BF16, INT8 };

constexpr size_t kSmePeakScratchElems = 4096;
constexpr size_t kSmePeakTileElems = 65536;
constexpr uint64_t kIosSmePeakInner = 1ull << 25;

mach_timebase_info_data_t& timebase() {
  static mach_timebase_info_data_t tb = [] {
    mach_timebase_info_data_t v;
    mach_timebase_info(&v);
    return v;
  }();
  return tb;
}

double ticks_to_seconds(uint64_t ticks) {
  const auto& tb = timebase();
  return static_cast<double>(ticks) * static_cast<double>(tb.numer) /
         static_cast<double>(tb.denom) / 1e9;
}

int cpu_thread_count() {
  const unsigned hc = std::thread::hardware_concurrency();
  return static_cast<int>(std::max(1u, hc));
}

int env_int_or_zero(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) return 0;
  char* end = nullptr;
  long parsed = std::strtol(v, &end, 10);
  if (end == v || parsed <= 0 || parsed > 1024) return 0;
  return static_cast<int>(parsed);
}

int cpu_sme_thread_count() {
  if (int v = env_int_or_zero("APPLEFLOPS_IOS_CPU_SME_THREADS")) return v;
  if (int v = env_int_or_zero("APPLEFLOPS_CPU_THREADS")) return v;
  return cpu_thread_count() * 8;
}

void set_cpu_sme_worker_qos() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

bool has_arm_feature(const char* name) {
  int value = 0;
  size_t len = sizeof(value);
  return sysctlbyname(name, &value, &len, nullptr, 0) == 0 && value != 0;
}

NSNumber* arm_feature_value(const char* name) {
  int value = 0;
  size_t len = sizeof(value);
  if (sysctlbyname(name, &value, &len, nullptr, 0) != 0) return @(-1);
  return @(value);
}

uint16_t float_to_bf16_bits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t lsb = (u >> 16) & 1u;
  const uint32_t bias = 0x7FFFu + lsb;
  return static_cast<uint16_t>((u + bias) >> 16);
}

template <typename T>
T* aligned_alloc_64(size_t count) {
  void* p = nullptr;
  const size_t bytes = count * sizeof(T);
  if (posix_memalign(&p, 64, bytes) != 0 || p == nullptr) return nullptr;
  std::memset(p, 0, bytes);
  return static_cast<T*>(p);
}

template <typename Fn>
double best_seconds(int warmup, int repeats, Fn&& fn) {
  for (int i = 0; i < warmup; i++) fn();
  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < repeats; i++) {
    const uint64_t t0 = mach_absolute_time();
    fn();
    const uint64_t t1 = mach_absolute_time();
    best = std::min(best, ticks_to_seconds(t1 - t0));
  }
  return best;
}

NSDictionary* make_row(NSString* unit, NSString* precision, double score, NSString* metric,
                       double seconds, NSString* note) {
  return @{
    @"unit" : unit,
    @"precision" : precision,
    @"score" : @(score),
    @"metric" : metric,
    @"seconds" : @(seconds),
    @"note" : note ?: @""
  };
}

NSDictionary* make_error_row(NSString* unit, NSString* precision, NSString* error) {
  return @{
    @"unit" : unit,
    @"precision" : precision,
    @"score" : @0.0,
    @"metric" : @"-",
    @"seconds" : @0.0,
    @"note" : error ?: @"failed"
  };
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme"), noinline))
uint64_t cpu_sme_smoke_worker() {
  svbool_t pg = svptrue_b32();
  svfloat32_t x = svdup_n_f32(1.0f);
  svzero_za();
  svmopa_za32_m(0, pg, pg, x, x);
  return svcntsw() * sizeof(float);
}

void sme_probe_signal_handler(int sig) {
  g_sme_probe_signal = sig;
  siglongjmp(g_sme_probe_jmp, 1);
}

NSDictionary* run_sme_probe() {
  NSMutableDictionary* features = [NSMutableDictionary dictionary];
  const char* keys[] = {
      "hw.optional.arm.FEAT_SME",
      "hw.optional.arm.FEAT_SME2",
      "hw.optional.arm.FEAT_SME2p1",
      "hw.optional.arm.SME_I8I32",
      "hw.optional.arm.FEAT_SME_F16F16",
      "hw.optional.arm.FEAT_SME_B16B16",
      "hw.optional.arm.FEAT_SME_F64F64",
      "hw.optional.arm.FEAT_SME_I16I64",
  };
  for (const char* key : keys) {
    features[[NSString stringWithUTF8String:key]] = arm_feature_value(key);
  }

  if (!has_arm_feature("hw.optional.arm.FEAT_SME")) {
    return @{
      @"features" : features,
      @"executed" : @NO,
      @"signal" : @0,
      @"svlBytes" : @0,
      @"note" : @"hw.optional.arm.FEAT_SME is not available to this process"
    };
  }

  struct sigaction old_ill {};
  struct sigaction old_bus {};
  struct sigaction old_segv {};
  struct sigaction old_trap {};
  struct sigaction act {};
  act.sa_handler = sme_probe_signal_handler;
  sigemptyset(&act.sa_mask);
  sigaction(SIGILL, &act, &old_ill);
  sigaction(SIGBUS, &act, &old_bus);
  sigaction(SIGSEGV, &act, &old_segv);
  sigaction(SIGTRAP, &act, &old_trap);

  g_sme_probe_signal = 0;
  uint64_t svl = 0;
  bool ok = false;
  if (sigsetjmp(g_sme_probe_jmp, 1) == 0) {
    svl = cpu_sme_smoke_worker();
    ok = true;
  }

  sigaction(SIGILL, &old_ill, nullptr);
  sigaction(SIGBUS, &old_bus, nullptr);
  sigaction(SIGSEGV, &old_segv, nullptr);
  sigaction(SIGTRAP, &old_trap, nullptr);

  return @{
    @"features" : features,
    @"executed" : @(ok),
    @"signal" : @((int)g_sme_probe_signal),
    @"svlBytes" : @(svl),
    @"note" : ok ? @"SME F32 MOPA smoke executed" : @"SME smoke trapped"
  };
}

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
  *ops_per_mopa = 4ull * n32 * n32;
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
  *ops_per_mopa = 4ull * n32 * n32;
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
  *ops_per_mopa = 8ull * n32 * n32;
  svzero_za();
  for (uint64_t i = 0; i < inner; i++) {
    svmopa_za32_m(0, pg8, pg8, va, vb);
  }
  for (uint32_t r = 0; r < n32; r++) {
    svst1_hor_za32(0, r, pg32, out + static_cast<size_t>(r) * n32);
  }
}

NSString* cpu_sme_required_feature(CpuPrecision p) {
  switch (p) {
    case CpuPrecision::FP32: return @"hw.optional.arm.FEAT_SME";
    case CpuPrecision::FP16: return @"hw.optional.arm.FEAT_SME_F16F16";
    case CpuPrecision::BF16: return @"hw.optional.arm.FEAT_SME_B16B16";
    case CpuPrecision::INT8: return @"hw.optional.arm.SME_I8I32";
  }
}

NSString* cpu_sme_note(CpuPrecision p, uint64_t inner, int threads, uint32_t lanes32) {
  NSString* accumulate = (p == CpuPrecision::INT8) ? @"int32 accumulate" : @"fp32 accumulate";
  return [NSString stringWithFormat:@"sme_mopa peak_probe | inner=%llu | threads=%d | svl=%uB | %@",
                                    static_cast<unsigned long long>(inner), threads,
                                    lanes32 * static_cast<uint32_t>(sizeof(float)), accumulate];
}

NSDictionary* run_cpu_sme_peak_one(CpuPrecision p, NSString* name, NSString* metric) {
  NSString* feature = cpu_sme_required_feature(p);
  if (!has_arm_feature([feature UTF8String])) {
    return make_error_row(@"CPU SME", name,
                          [NSString stringWithFormat:@"%@ is not available", feature]);
  }

  const uint64_t inner = kIosSmePeakInner;
  const int threads = cpu_sme_thread_count();
  uint64_t ops_per_mopa = 0;
  uint32_t lanes32 = 0;

  if (p == CpuPrecision::FP32) {
    float* a = aligned_alloc_64<float>(kSmePeakScratchElems);
    float* b = aligned_alloc_64<float>(kSmePeakScratchElems);
    if (!a || !b) {
      std::free(a);
      std::free(b);
      return make_error_row(@"CPU SME", name, @"posix_memalign failed");
    }
    std::vector<float*> outs(static_cast<size_t>(threads), nullptr);
    for (size_t i = 0; i < kSmePeakScratchElems; i++) a[i] = b[i] = 1.0f;
    for (float*& out : outs) {
      out = aligned_alloc_64<float>(kSmePeakTileElems);
      if (!out) {
        std::free(a);
        std::free(b);
        for (float* p : outs) std::free(p);
        return make_error_row(@"CPU SME", name, @"posix_memalign failed");
      }
    }
    auto run_parallel = [&] {
      std::vector<std::thread> workers;
      std::vector<uint64_t> ops(static_cast<size_t>(threads), 0);
      std::vector<uint32_t> lanes(static_cast<size_t>(threads), 0);
      workers.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; t++) {
        workers.emplace_back([&, t] {
          set_cpu_sme_worker_qos();
          cpu_sme_peak_fp32_kernel(a, b, outs[t], inner, &ops[t], &lanes[t]);
        });
      }
      for (auto& w : workers) w.join();
      ops_per_mopa = ops[0];
      lanes32 = lanes[0];
    };
    const double sec = best_seconds(1, 3, run_parallel);
    volatile float sink = outs[0][0];
    (void)sink;
    std::free(a);
    std::free(b);
    for (float* out : outs) std::free(out);
    const double ops = static_cast<double>(ops_per_mopa) * static_cast<double>(inner) *
                       static_cast<double>(threads);
    return make_row(@"CPU SME", name, ops / sec / 1e12, metric, sec,
                    cpu_sme_note(p, inner, threads, lanes32));
  }

  if (p == CpuPrecision::FP16) {
    __fp16* a = aligned_alloc_64<__fp16>(kSmePeakScratchElems);
    __fp16* b = aligned_alloc_64<__fp16>(kSmePeakScratchElems);
    if (!a || !b) {
      std::free(a);
      std::free(b);
      return make_error_row(@"CPU SME", name, @"posix_memalign failed");
    }
    std::vector<float*> outs(static_cast<size_t>(threads), nullptr);
    for (size_t i = 0; i < kSmePeakScratchElems; i++) {
      a[i] = static_cast<__fp16>(1.0f);
      b[i] = static_cast<__fp16>(1.0f);
    }
    for (float*& out : outs) {
      out = aligned_alloc_64<float>(kSmePeakTileElems);
      if (!out) {
        std::free(a);
        std::free(b);
        for (float* p : outs) std::free(p);
        return make_error_row(@"CPU SME", name, @"posix_memalign failed");
      }
    }
    auto run_parallel = [&] {
      std::vector<std::thread> workers;
      std::vector<uint64_t> ops(static_cast<size_t>(threads), 0);
      std::vector<uint32_t> lanes(static_cast<size_t>(threads), 0);
      workers.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; t++) {
        workers.emplace_back([&, t] {
          set_cpu_sme_worker_qos();
          cpu_sme_peak_fp16_kernel(a, b, outs[t], inner, &ops[t], &lanes[t]);
        });
      }
      for (auto& w : workers) w.join();
      ops_per_mopa = ops[0];
      lanes32 = lanes[0];
    };
    const double sec = best_seconds(1, 3, run_parallel);
    volatile float sink = outs[0][0];
    (void)sink;
    std::free(a);
    std::free(b);
    for (float* out : outs) std::free(out);
    const double ops = static_cast<double>(ops_per_mopa) * static_cast<double>(inner) *
                       static_cast<double>(threads);
    return make_row(@"CPU SME", name, ops / sec / 1e12, metric, sec,
                    cpu_sme_note(p, inner, threads, lanes32));
  }

  if (p == CpuPrecision::BF16) {
    bfloat16_t* a = aligned_alloc_64<bfloat16_t>(kSmePeakScratchElems);
    bfloat16_t* b = aligned_alloc_64<bfloat16_t>(kSmePeakScratchElems);
    if (!a || !b) {
      std::free(a);
      std::free(b);
      return make_error_row(@"CPU SME", name, @"posix_memalign failed");
    }
    std::vector<float*> outs(static_cast<size_t>(threads), nullptr);
    const uint16_t one_bits = float_to_bf16_bits(1.0f);
    for (size_t i = 0; i < kSmePeakScratchElems; i++) {
      std::memcpy(&a[i], &one_bits, sizeof(one_bits));
      std::memcpy(&b[i], &one_bits, sizeof(one_bits));
    }
    for (float*& out : outs) {
      out = aligned_alloc_64<float>(kSmePeakTileElems);
      if (!out) {
        std::free(a);
        std::free(b);
        for (float* p : outs) std::free(p);
        return make_error_row(@"CPU SME", name, @"posix_memalign failed");
      }
    }
    auto run_parallel = [&] {
      std::vector<std::thread> workers;
      std::vector<uint64_t> ops(static_cast<size_t>(threads), 0);
      std::vector<uint32_t> lanes(static_cast<size_t>(threads), 0);
      workers.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; t++) {
        workers.emplace_back([&, t] {
          set_cpu_sme_worker_qos();
          cpu_sme_peak_bf16_kernel(a, b, outs[t], inner, &ops[t], &lanes[t]);
        });
      }
      for (auto& w : workers) w.join();
      ops_per_mopa = ops[0];
      lanes32 = lanes[0];
    };
    const double sec = best_seconds(1, 3, run_parallel);
    volatile float sink = outs[0][0];
    (void)sink;
    std::free(a);
    std::free(b);
    for (float* out : outs) std::free(out);
    const double ops = static_cast<double>(ops_per_mopa) * static_cast<double>(inner) *
                       static_cast<double>(threads);
    return make_row(@"CPU SME", name, ops / sec / 1e12, metric, sec,
                    cpu_sme_note(p, inner, threads, lanes32));
  }

  int8_t* a = aligned_alloc_64<int8_t>(kSmePeakScratchElems);
  int8_t* b = aligned_alloc_64<int8_t>(kSmePeakScratchElems);
  if (!a || !b) {
    std::free(a);
    std::free(b);
    return make_error_row(@"CPU SME", name, @"posix_memalign failed");
  }
  std::vector<int32_t*> outs(static_cast<size_t>(threads), nullptr);
  for (size_t i = 0; i < kSmePeakScratchElems; i++) a[i] = b[i] = 1;
  for (int32_t*& out : outs) {
    out = aligned_alloc_64<int32_t>(kSmePeakTileElems);
    if (!out) {
      std::free(a);
      std::free(b);
      for (int32_t* p : outs) std::free(p);
      return make_error_row(@"CPU SME", name, @"posix_memalign failed");
    }
  }
  auto run_parallel = [&] {
    std::vector<std::thread> workers;
    std::vector<uint64_t> ops(static_cast<size_t>(threads), 0);
    std::vector<uint32_t> lanes(static_cast<size_t>(threads), 0);
    workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; t++) {
      workers.emplace_back([&, t] {
        set_cpu_sme_worker_qos();
        cpu_sme_peak_int8_kernel(a, b, outs[t], inner, &ops[t], &lanes[t]);
      });
    }
    for (auto& w : workers) w.join();
    ops_per_mopa = ops[0];
    lanes32 = lanes[0];
  };
  const double sec = best_seconds(1, 3, run_parallel);
  volatile int32_t sink = outs[0][0];
  (void)sink;
  std::free(a);
  std::free(b);
  for (int32_t* out : outs) std::free(out);
  const double ops = static_cast<double>(ops_per_mopa) * static_cast<double>(inner) *
                     static_cast<double>(threads);
  return make_row(@"CPU SME", name, ops / sec / 1e12, metric, sec,
                  cpu_sme_note(p, inner, threads, lanes32));
}

NSArray* run_cpu_sme_benches(AppleFLOPSProgressBlock progress) {
  NSMutableArray* rows = [NSMutableArray array];
  if (progress) progress(@"CPU SME FP32");
  [rows addObject:run_cpu_sme_peak_one(CpuPrecision::FP32, @"FP32", @"TFLOPS")];
  if (progress) progress(@"CPU SME FP16");
  [rows addObject:run_cpu_sme_peak_one(CpuPrecision::FP16, @"FP16", @"TFLOPS")];
  if (progress) progress(@"CPU SME BF16");
  [rows addObject:run_cpu_sme_peak_one(CpuPrecision::BF16, @"BF16", @"TFLOPS")];
  if (progress) progress(@"CPU SME INT8");
  [rows addObject:run_cpu_sme_peak_one(CpuPrecision::INT8, @"INT8", @"TOPS")];
  return rows;
}

__attribute__((target("fullfp16")))
void cpu_fp16_worker(uint64_t iters) {
  float16x8_t a = vdupq_n_f16(static_cast<__fp16>(1.001f));
  float16x8_t b = vdupq_n_f16(static_cast<__fp16>(1.003f));
  float16x8_t c0 = vdupq_n_f16(static_cast<__fp16>(0.1f));
  float16x8_t c1 = vdupq_n_f16(static_cast<__fp16>(0.2f));
  float16x8_t c2 = vdupq_n_f16(static_cast<__fp16>(0.3f));
  float16x8_t c3 = vdupq_n_f16(static_cast<__fp16>(0.4f));
  for (uint64_t i = 0; i < iters; i++) {
    c0 = vfmaq_f16(c0, a, b);
    c1 = vfmaq_f16(c1, a, b);
    c2 = vfmaq_f16(c2, a, b);
    c3 = vfmaq_f16(c3, a, b);
  }
  alignas(16) __fp16 tmp[8];
  vst1q_f16(tmp, vaddq_f16(vaddq_f16(c0, c1), vaddq_f16(c2, c3)));
  volatile float sink = static_cast<float>(tmp[0]);
  (void)sink;
}

__attribute__((target("bf16")))
void cpu_bf16_worker(uint64_t iters) {
  bfloat16x8_t a = vreinterpretq_bf16_u16(vdupq_n_u16(float_to_bf16_bits(1.001f)));
  bfloat16x8_t b = vreinterpretq_bf16_u16(vdupq_n_u16(float_to_bf16_bits(1.003f)));
  float32x4_t c0 = vdupq_n_f32(0.1f);
  float32x4_t c1 = vdupq_n_f32(0.2f);
  float32x4_t c2 = vdupq_n_f32(0.3f);
  float32x4_t c3 = vdupq_n_f32(0.4f);
  for (uint64_t i = 0; i < iters; i++) {
    c0 = vbfmmlaq_f32(c0, a, b);
    c1 = vbfmmlaq_f32(c1, a, b);
    c2 = vbfmmlaq_f32(c2, a, b);
    c3 = vbfmmlaq_f32(c3, a, b);
  }
  volatile float sink = vaddvq_f32(vaddq_f32(vaddq_f32(c0, c1), vaddq_f32(c2, c3)));
  (void)sink;
}

__attribute__((target("i8mm")))
void cpu_int8_i8mm_worker(uint64_t iters) {
  int8x16_t a = vdupq_n_s8(3);
  int8x16_t b = vdupq_n_s8(5);
  int32x4_t c0 = vdupq_n_s32(0);
  int32x4_t c1 = vdupq_n_s32(1);
  int32x4_t c2 = vdupq_n_s32(2);
  int32x4_t c3 = vdupq_n_s32(3);
  for (uint64_t i = 0; i < iters; i++) {
    c0 = vmmlaq_s32(c0, a, b);
    c1 = vmmlaq_s32(c1, a, b);
    c2 = vmmlaq_s32(c2, a, b);
    c3 = vmmlaq_s32(c3, a, b);
  }
  volatile int32_t sink = vaddvq_s32(vaddq_s32(vaddq_s32(c0, c1), vaddq_s32(c2, c3)));
  (void)sink;
}

__attribute__((target("dotprod")))
void cpu_int8_dotprod_worker(uint64_t iters) {
  int8x16_t a = vdupq_n_s8(3);
  int8x16_t b = vdupq_n_s8(5);
  int32x4_t c0 = vdupq_n_s32(0);
  int32x4_t c1 = vdupq_n_s32(1);
  int32x4_t c2 = vdupq_n_s32(2);
  int32x4_t c3 = vdupq_n_s32(3);
  for (uint64_t i = 0; i < iters; i++) {
    c0 = vdotq_s32(c0, a, b);
    c1 = vdotq_s32(c1, a, b);
    c2 = vdotq_s32(c2, a, b);
    c3 = vdotq_s32(c3, a, b);
  }
  volatile int32_t sink = vaddvq_s32(vaddq_s32(vaddq_s32(c0, c1), vaddq_s32(c2, c3)));
  (void)sink;
}

NSDictionary* run_cpu_fp32() {
  constexpr int lanes = 4;
  constexpr uint64_t iters = 1ull << 26;
  const int threads = cpu_thread_count();
  auto worker = [] {
    float32x4_t a = vdupq_n_f32(1.001f);
    float32x4_t b = vdupq_n_f32(1.003f);
    float32x4_t c0 = vdupq_n_f32(0.1f);
    float32x4_t c1 = vdupq_n_f32(0.2f);
    float32x4_t c2 = vdupq_n_f32(0.3f);
    float32x4_t c3 = vdupq_n_f32(0.4f);
    for (uint64_t i = 0; i < iters; i++) {
      c0 = vfmaq_f32(c0, a, b);
      c1 = vfmaq_f32(c1, a, b);
      c2 = vfmaq_f32(c2, a, b);
      c3 = vfmaq_f32(c3, a, b);
    }
    volatile float sink = vaddvq_f32(vaddq_f32(vaddq_f32(c0, c1), vaddq_f32(c2, c3)));
    (void)sink;
  };
  const double sec = best_seconds(1, 3, [&] {
    std::vector<std::thread> ws;
    ws.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; t++) ws.emplace_back(worker);
    for (auto& w : ws) w.join();
  });
  const double ops = static_cast<double>(threads) * static_cast<double>(iters) * 4.0 *
                     static_cast<double>(lanes) * 2.0;
  return make_row(@"CPU NEON", @"FP32", ops / sec / 1e12, @"TFLOPS", sec,
                  [NSString stringWithFormat:@"neon_fma_peak threads=%d", threads]);
}

NSDictionary* run_cpu_fp16() {
  if (!has_arm_feature("hw.optional.arm.FEAT_FP16")) {
    return make_error_row(@"CPU NEON", @"FP16", @"hw.optional.arm.FEAT_FP16 is not available");
  }
  constexpr int lanes = 8;
  constexpr uint64_t iters = 1ull << 26;
  const int threads = cpu_thread_count();
  const double sec = best_seconds(1, 3, [&] {
    std::vector<std::thread> ws;
    ws.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; t++) ws.emplace_back([] { cpu_fp16_worker(iters); });
    for (auto& w : ws) w.join();
  });
  const double ops = static_cast<double>(threads) * static_cast<double>(iters) * 4.0 *
                     static_cast<double>(lanes) * 2.0;
  return make_row(@"CPU NEON", @"FP16", ops / sec / 1e12, @"TFLOPS", sec,
                  [NSString stringWithFormat:@"neon_fp16_fma_peak threads=%d", threads]);
}

NSDictionary* run_cpu_bf16() {
  if (!has_arm_feature("hw.optional.arm.FEAT_BF16")) {
    return make_error_row(@"CPU NEON", @"BF16", @"hw.optional.arm.FEAT_BF16 is not available");
  }
  constexpr uint64_t iters = 1ull << 26;
  const int threads = cpu_thread_count();
  const double sec = best_seconds(1, 3, [&] {
    std::vector<std::thread> ws;
    ws.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; t++) ws.emplace_back([] { cpu_bf16_worker(iters); });
    for (auto& w : ws) w.join();
  });
  const double ops = static_cast<double>(threads) * static_cast<double>(iters) * 4.0 * 16.0 * 2.0;
  return make_row(@"CPU NEON", @"BF16", ops / sec / 1e12, @"TFLOPS", sec,
                  [NSString stringWithFormat:@"neon_bf16_mmla_peak threads=%d", threads]);
}

NSDictionary* run_cpu_int8() {
  constexpr uint64_t iters = 1ull << 25;
  const int threads = cpu_thread_count();
  if (has_arm_feature("hw.optional.arm.FEAT_I8MM")) {
    const double sec = best_seconds(1, 3, [&] {
      std::vector<std::thread> ws;
      ws.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; t++) ws.emplace_back([] { cpu_int8_i8mm_worker(iters); });
      for (auto& w : ws) w.join();
    });
    const double ops = static_cast<double>(threads) * static_cast<double>(iters) * 4.0 * 32.0 * 2.0;
    return make_row(@"CPU NEON", @"INT8", ops / sec / 1e12, @"TOPS", sec,
                    [NSString stringWithFormat:@"neon_i8mm_peak threads=%d", threads]);
  }
  if (has_arm_feature("hw.optional.arm.FEAT_DotProd")) {
    const double sec = best_seconds(1, 3, [&] {
      std::vector<std::thread> ws;
      ws.reserve(static_cast<size_t>(threads));
      for (int t = 0; t < threads; t++) ws.emplace_back([] { cpu_int8_dotprod_worker(iters); });
      for (auto& w : ws) w.join();
    });
    const double ops = static_cast<double>(threads) * static_cast<double>(iters) * 4.0 * 16.0 * 2.0;
    return make_row(@"CPU NEON", @"INT8", ops / sec / 1e12, @"TOPS", sec,
                    [NSString stringWithFormat:@"neon_dotprod_peak threads=%d", threads]);
  }
  return make_error_row(@"CPU NEON", @"INT8", @"hw.optional.arm.FEAT_I8MM/DotProd is not available");
}

NSArray* run_cpu_benches(AppleFLOPSProgressBlock progress) {
  if (has_arm_feature("hw.optional.arm.FEAT_SME")) {
    return run_cpu_sme_benches(progress);
  }
  if (progress) progress(@"CPU FP32");
  NSMutableArray* rows = [NSMutableArray array];
  [rows addObject:run_cpu_fp32()];
  if (progress) progress(@"CPU FP16");
  [rows addObject:run_cpu_fp16()];
  if (progress) progress(@"CPU BF16");
  [rows addObject:run_cpu_bf16()];
  if (progress) progress(@"CPU INT8");
  [rows addObject:run_cpu_int8()];
  return rows;
}

NSDictionary* run_gpu_one(GpuPrecision p, NSString* name, NSString* shaderPath) {
  GpuBenchOptions opt;
  opt.n = 1024;
  opt.warmup = 1;
  opt.repeats = 3;
  opt.precision = p;
  opt.shader_path = std::string([shaderPath UTF8String]);
  opt.batch = 8;
  opt.inner = 256;
  opt.storage = GpuStorageMode::Private;
  opt.workload = GpuWorkload::Peak;
  GpuBenchResult out;
  std::string err;
  if (!run_gpu_bench(opt, out, err)) {
    return make_error_row(@"GPU (simdgroup)", name, [NSString stringWithUTF8String:err.c_str()]);
  }
  return make_row(@"GPU (simdgroup)", name, out.score, @"TFLOPS", out.best_seconds,
                  [NSString stringWithFormat:@"backend=%s n=%d inner=%d batch=%d",
                                             out.backend.c_str(), opt.n, opt.inner, opt.batch]);
}

NSArray* run_gpu_benches(AppleFLOPSProgressBlock progress) {
  NSString* shaderPath = [[NSBundle mainBundle] pathForResource:@"gemm" ofType:@"metal"];
  NSMutableArray* rows = [NSMutableArray array];
  if (!shaderPath) {
    [rows addObject:make_error_row(@"GPU (simdgroup)", @"FP32", @"gemm.metal not found in app bundle")];
    return rows;
  }
  if (progress) progress(@"GPU FP16");
  [rows addObject:run_gpu_one(GpuPrecision::FP16, @"FP16", shaderPath)];
  if (progress) progress(@"GPU FP32");
  [rows addObject:run_gpu_one(GpuPrecision::FP32, @"FP32", shaderPath)];
  if (progress) progress(@"GPU BF16");
  [rows addObject:run_gpu_one(GpuPrecision::BF16, @"BF16", shaderPath)];
  [rows addObject:make_error_row(@"GPU", @"INT8", @"GPU INT8 is not recorded as trusted matrix-unit peak")];
  return rows;
}

NSDictionary* run_npu_one(NpuPrecision p, NSString* name, NSString* metric) {
  NpuBenchOptions opt;
  opt.channels = 512;
  opt.spatial = 64;
  opt.depth = 128;
  opt.warmup = 3;
  opt.repeats = 5;
  opt.precision = p;
  NpuBenchResult out;
  std::string err;
  if (!run_npu_bench(opt, out, err)) {
    return make_error_row(@"NPU (private ANE)", name, [NSString stringWithUTF8String:err.c_str()]);
  }
  return make_row(@"NPU (private ANE)", name, out.score, metric, out.best_seconds,
                  [NSString stringWithFormat:@"%s | %s", out.backend.c_str(), out.note.c_str()]);
}

NSArray* run_npu_benches(AppleFLOPSProgressBlock progress) {
  NSMutableArray* rows = [NSMutableArray array];
  if (progress) progress(@"NPU FP16");
  [rows addObject:run_npu_one(NpuPrecision::FP16, @"FP16", @"TFLOPS")];
  if (progress) progress(@"NPU INT8");
  [rows addObject:run_npu_one(NpuPrecision::INT8, @"INT8", @"TOPS")];
  [rows addObject:make_error_row(@"NPU (private ANE)", @"FP32", @"NPU FP32 is disabled")];
  [rows addObject:make_error_row(@"NPU (private ANE)", @"BF16", @"NPU BF16 is not recorded as native score")];
  return rows;
}

}  // namespace

NSDictionary* RunIOSBenchmarks(AppleFLOPSProgressBlock progress) {
  NSMutableArray* rows = [NSMutableArray array];
  if (progress) progress(@"Starting CPU");
  [rows addObjectsFromArray:run_cpu_benches(progress)];
  if (progress) progress(@"Starting GPU");
  [rows addObjectsFromArray:run_gpu_benches(progress)];
  if (progress) progress(@"Starting NPU");
  [rows addObjectsFromArray:run_npu_benches(progress)];

  NSString* model = @"unknown";
  size_t len = 0;
  if (sysctlbyname("hw.machine", nullptr, &len, nullptr, 0) == 0 && len > 0) {
    std::vector<char> buf(len);
    if (sysctlbyname("hw.machine", buf.data(), &len, nullptr, 0) == 0) {
      model = [NSString stringWithUTF8String:buf.data()];
    }
  }

  return @{
    @"tool" : @"AppleFLOPS iOS Runner",
    @"timestamp" : @([[NSDate date] timeIntervalSince1970]),
    @"device" : @{
      @"name" : @"redacted",
      @"model" : model,
      @"systemName" : [[UIDevice currentDevice] systemName],
      @"systemVersion" : [[UIDevice currentDevice] systemVersion]
    },
    @"smeProbe" : run_sme_probe(),
    @"rows" : rows
  };
}

NSString* FormatBenchmarkReport(NSDictionary* result) {
  NSMutableString* s = [NSMutableString string];
  NSDictionary* device = result[@"device"];
  [s appendFormat:@"AppleFLOPS iOS Runner\n%@ %@ (%@)\n\n",
                  device[@"systemName"], device[@"systemVersion"], device[@"model"]];
  [s appendFormat:@"%-18s %-6s %10s %-7s %8s  %s\n",
                  "Unit", "Prec", "Score", "Metric", "ms", "Note"];
  for (NSDictionary* row in result[@"rows"]) {
    const double score = [row[@"score"] doubleValue];
    const double ms = [row[@"seconds"] doubleValue] * 1000.0;
    [s appendFormat:@"%-18s %-6s %10.3f %-7s %8.3f  %s\n",
                    [row[@"unit"] UTF8String], [row[@"precision"] UTF8String],
                    score, [row[@"metric"] UTF8String], ms, [row[@"note"] UTF8String]];
  }
  return s;
}

NSURL* WriteBenchmarkResult(NSDictionary* result, NSError** error) {
  NSData* data = [NSJSONSerialization dataWithJSONObject:result
                                                 options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                   error:error];
  if (!data) return nil;
  NSURL* docs = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory
                                                       inDomains:NSUserDomainMask] firstObject];
  NSURL* url = [docs URLByAppendingPathComponent:@"appleflops-ios-results.json"];
  if (![data writeToURL:url options:NSDataWritingAtomic error:error]) return nil;
  return url;
}
