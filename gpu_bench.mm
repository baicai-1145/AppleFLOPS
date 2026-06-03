#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu_bench.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sys/stat.h>
#include <vector>

namespace {

double tflops_for_gemm(int n, double seconds) {
  const double nn = static_cast<double>(n);
  const double flops = 2.0 * nn * nn * nn;
  return flops / seconds / 1e12;
}

double score_for_gemm(int n, int batch, int inner, double seconds) {
  return static_cast<double>(batch) * static_cast<double>(inner) * tflops_for_gemm(n, seconds);
}

double score_for_peak(GpuPrecision p, int n, int batch, int inner, double seconds) {
  // peak workload:
  // - grid: (n/8) x (n/8) threadgroups
  // - FP/BF16: each threadgroup runs 4 simdgroups, each with 4 accumulators
  //   * inner multiplies
  // - each 8x8x8 MMA = 1024 FLOPs
  // INT8 peak uses a scalar 32-thread probe, so every thread contributes the same
  // 4*512 MAC operations. GEMM INT8 remains the more representative workload.
  const double tg = static_cast<double>(n / 8) * static_cast<double>(n / 8);
  const double per_tg = (p == GpuPrecision::INT8) ? (32.0 * 4.0 * 1024.0) : (4.0 * 4.0 * 1024.0);
  const double flops = tg * static_cast<double>(batch) * static_cast<double>(inner) * per_tg;
  return flops / seconds / 1e12;
}
static const char* kernel_name(GpuPrecision p, GpuKernelVariant k) {
  switch (k) {
    case GpuKernelVariant::V4:
      if (p == GpuPrecision::FP32) return "gemm_fp32_v4";
      if (p == GpuPrecision::FP16) return "gemm_fp16_v4";
      if (p == GpuPrecision::BF16) return "gemm_bf16_v4";
      return "gemm_int8_v4";
    case GpuKernelVariant::Auto:
      if (p == GpuPrecision::FP32) return "gemm_fp32_v4";
      if (p == GpuPrecision::FP16) return "gemm_fp16_v4";
      if (p == GpuPrecision::BF16) return "gemm_bf16_v4";
      return "gemm_int8_v4";
  }
  if (p == GpuPrecision::FP32) return "gemm_fp32_v4";
  if (p == GpuPrecision::FP16) return "gemm_fp16_v4";
  if (p == GpuPrecision::BF16) return "gemm_bf16_v4";
  return "gemm_int8_v4";
}

static bool kernel_supported(int n, GpuPrecision p, GpuKernelVariant k) {
  if (p == GpuPrecision::INT8) return n > 0;
  if (k == GpuKernelVariant::V4) return (n % 32) == 0;
  // Auto：发布版仅保留 v4。
  return (n % 32) == 0;
}

static bool file_mtime(const std::string& path, time_t& out) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return false;
  out = st.st_mtime;
  return true;
}

static void fill_fp32(float* p, int n, uint32_t seed) {
  uint32_t x = seed ? seed : 1u;
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < nn; i++) {
    x = x * 1664525u + 1013904223u;
    p[i] = static_cast<float>((x >> 8) & 0x00FFFFFF) / static_cast<float>(0x01000000);
  }
}

static void fill_fp16(__fp16* p, int n, uint32_t seed) {
  uint32_t x = seed ? seed : 1u;
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < nn; i++) {
    x = x * 1664525u + 1013904223u;
    const float f = static_cast<float>((x >> 8) & 0x00FFFFFF) / static_cast<float>(0x01000000);
    p[i] = static_cast<__fp16>(f);
  }
}

static uint16_t float_to_bf16(float f) {
  uint32_t u = 0;
  static_assert(sizeof(float) == sizeof(uint32_t));
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t lsb = (u >> 16) & 1u;
  const uint32_t bias = 0x7FFFu + lsb;  // round-to-nearest-even
  return static_cast<uint16_t>((u + bias) >> 16);
}

static void fill_bf16(uint16_t* p, int n, uint32_t seed) {
  uint32_t x = seed ? seed : 1u;
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < nn; i++) {
    x = x * 1664525u + 1013904223u;
    const float f = static_cast<float>((x >> 8) & 0x00FFFFFF) / static_cast<float>(0x01000000);
    p[i] = float_to_bf16(f);
  }
}

static void fill_int8(int8_t* p, int n, uint32_t seed) {
  uint32_t x = seed ? seed : 1u;
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < nn; i++) {
    x = x * 1664525u + 1013904223u;
    p[i] = static_cast<int8_t>(((x >> 24) & 0x0F) - 8);
  }
}

static NSString* ns_error(NSError* e) {
  if (!e) return @"(nil)";
  return [NSString stringWithFormat:@"%@ (code=%ld)", e.localizedDescription, (long)e.code];
}

struct MetalCache {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  std::string shader_path;
  time_t shader_mtime = 0;
  id<MTLLibrary> lib = nil;
  id<MTLComputePipelineState> pso_fp32_v4 = nil;
  id<MTLComputePipelineState> pso_fp16_v4 = nil;
  id<MTLComputePipelineState> pso_bf16_v4 = nil;
  id<MTLComputePipelineState> pso_int8_v4 = nil;
  id<MTLComputePipelineState> pso_peak_fp32 = nil;
  id<MTLComputePipelineState> pso_peak_fp16 = nil;
  id<MTLComputePipelineState> pso_peak_bf16 = nil;
  id<MTLComputePipelineState> pso_peak_int8 = nil;
};

static MetalCache& cache() {
  static MetalCache c;
  return c;
}

static id<MTLComputePipelineState> cached_pso(GpuPrecision p, GpuKernelVariant k) {
  auto& c = cache();
  if (p == GpuPrecision::FP32 && k == GpuKernelVariant::V4) return c.pso_fp32_v4;
  if (p == GpuPrecision::FP16 && k == GpuKernelVariant::V4) return c.pso_fp16_v4;
  if (p == GpuPrecision::BF16 && k == GpuKernelVariant::V4) return c.pso_bf16_v4;
  if (p == GpuPrecision::INT8 && k == GpuKernelVariant::V4) return c.pso_int8_v4;
  return nil;
}

static void set_cached_pso(GpuPrecision p, GpuKernelVariant k, id<MTLComputePipelineState> pso) {
  auto& c = cache();
  if (p == GpuPrecision::FP32 && k == GpuKernelVariant::V4) c.pso_fp32_v4 = pso;
  else if (p == GpuPrecision::FP16 && k == GpuKernelVariant::V4) c.pso_fp16_v4 = pso;
  else if (p == GpuPrecision::BF16 && k == GpuKernelVariant::V4) c.pso_bf16_v4 = pso;
  else if (p == GpuPrecision::INT8 && k == GpuKernelVariant::V4) c.pso_int8_v4 = pso;
}

static id<MTLComputePipelineState> cached_peak_pso(GpuPrecision p) {
  auto& c = cache();
  if (p == GpuPrecision::FP32) return c.pso_peak_fp32;
  if (p == GpuPrecision::FP16) return c.pso_peak_fp16;
  if (p == GpuPrecision::BF16) return c.pso_peak_bf16;
  return c.pso_peak_int8;
}

static void set_cached_peak_pso(GpuPrecision p, id<MTLComputePipelineState> pso) {
  auto& c = cache();
  if (p == GpuPrecision::FP32) c.pso_peak_fp32 = pso;
  else if (p == GpuPrecision::FP16) c.pso_peak_fp16 = pso;
  else if (p == GpuPrecision::BF16) c.pso_peak_bf16 = pso;
  else c.pso_peak_int8 = pso;
}

}  // namespace

bool run_gpu_bench(const GpuBenchOptions& opt, GpuBenchResult& out, std::string& error) {
  @autoreleasepool {
    if (opt.n <= 0) {
      error = "n must be > 0";
      return false;
    }
    if (opt.workload == GpuWorkload::Gemm && !kernel_supported(opt.n, opt.precision, opt.kernel)) {
      error = "GPU kernel v4 requires N multiple of 32";
      return false;
    }
    if (opt.workload == GpuWorkload::Peak && (opt.n % 8) != 0) {
      error = "peak workload requires N multiple of 8";
      return false;
    }
    if (opt.repeats <= 0 || opt.warmup < 0) {
      error = "warmup must be >= 0 and repeats must be > 0";
      return false;
    }
    if (opt.batch <= 0) {
      error = "batch must be > 0";
      return false;
    }
    if (opt.inner <= 0) {
      error = "inner must be > 0";
      return false;
    }

    auto& c = cache();
    if (!c.device) {
      c.device = MTLCreateSystemDefaultDevice();
      if (!c.device) {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices.count > 0) c.device = devices[0];
      }
    }
    id<MTLDevice> device = c.device;
    if (!device) {
      error =
          "no Metal device available in current runtime (MTLCreateSystemDefaultDevice == nil). "
          "请在具备 Metal GPU 的本机 macOS 环境运行。";
      return false;
    }

    NSError* nsErr = nil;
    time_t mt = 0;
    if (!file_mtime(opt.shader_path, mt)) {
      error = "failed to stat shader source: " + opt.shader_path;
      return false;
    }
    const bool shader_changed = (c.shader_path != opt.shader_path) || (c.shader_mtime != mt) || (c.lib == nil);
    if (shader_changed) {
      NSString* path = [NSString stringWithUTF8String:opt.shader_path.c_str()];
      NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&nsErr];
      if (!src) {
        error = "failed to read shader source: " + std::string(opt.shader_path) +
                " (" + std::string([[ns_error(nsErr) description] UTF8String]) + ")";
        return false;
      }
      MTLCompileOptions* compileOpt = [[MTLCompileOptions alloc] init];
      if (@available(macOS 15.0, *)) {
        compileOpt.mathMode = MTLMathModeFast;
        compileOpt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
      } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        compileOpt.fastMathEnabled = YES;
#pragma clang diagnostic pop
      }
      c.lib = [device newLibraryWithSource:src options:compileOpt error:&nsErr];
      if (!c.lib) {
        error = "failed to compile metal shader: " + std::string([[ns_error(nsErr) description] UTF8String]);
        return false;
      }
      c.shader_path = opt.shader_path;
      c.shader_mtime = mt;
      c.pso_fp32_v4 = nil;
      c.pso_fp16_v4 = nil;
      c.pso_bf16_v4 = nil;
      c.pso_int8_v4 = nil;
      c.pso_peak_fp32 = c.pso_peak_fp16 = nil;
      c.pso_peak_bf16 = c.pso_peak_int8 = nil;
    }
    id<MTLLibrary> lib = c.lib;

    if (!c.queue) c.queue = [device newCommandQueue];
    id<MTLCommandQueue> queue = c.queue;
    if (!queue) {
      error = "failed to create MTLCommandQueue";
      return false;
    }

    const int n = opt.n;
    const size_t bytes_fp32 = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    const size_t bytes_fp16 = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(__fp16);
    const size_t bytes_bf16 = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(uint16_t);
    const size_t bytes_int8 = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(int8_t);

    auto input_bytes = [&]() -> size_t {
      if (opt.precision == GpuPrecision::FP32) return bytes_fp32;
      if (opt.precision == GpuPrecision::FP16) return bytes_fp16;
      if (opt.precision == GpuPrecision::BF16) return bytes_bf16;
      return bytes_int8;
    };

    const bool use_private = (opt.storage == GpuStorageMode::Private);
    id<MTLBuffer> bufA = nil;
    id<MTLBuffer> bufB = nil;
    id<MTLBuffer> bufC = nil;
    id<MTLBuffer> stageA = nil;
    id<MTLBuffer> stageB = nil;

    const MTLResourceOptions shared = MTLResourceStorageModeShared;
    const MTLResourceOptions priv = MTLResourceStorageModePrivate;

    if (opt.precision == GpuPrecision::FP32) {
      if (use_private) {
        stageA = [device newBufferWithLength:bytes_fp32 options:shared];
        stageB = [device newBufferWithLength:bytes_fp32 options:shared];
        bufA = [device newBufferWithLength:bytes_fp32 options:priv];
        bufB = [device newBufferWithLength:bytes_fp32 options:priv];
        bufC = [device newBufferWithLength:bytes_fp32 options:priv];
      } else {
        bufA = [device newBufferWithLength:bytes_fp32 options:shared];
        bufB = [device newBufferWithLength:bytes_fp32 options:shared];
        bufC = [device newBufferWithLength:bytes_fp32 options:shared];
      }
      if (!bufA || !bufB || !bufC || (use_private && (!stageA || !stageB))) {
        error = "failed to allocate MTLBuffers (fp32)";
        return false;
      }
      float* a = reinterpret_cast<float*>((use_private ? stageA : bufA).contents);
      float* b = reinterpret_cast<float*>((use_private ? stageB : bufB).contents);
      fill_fp32(a, n, 1);
      fill_fp32(b, n, 2);
    } else if (opt.precision == GpuPrecision::FP16) {
      if (use_private) {
        stageA = [device newBufferWithLength:bytes_fp16 options:shared];
        stageB = [device newBufferWithLength:bytes_fp16 options:shared];
        bufA = [device newBufferWithLength:bytes_fp16 options:priv];
        bufB = [device newBufferWithLength:bytes_fp16 options:priv];
        bufC = [device newBufferWithLength:bytes_fp32 options:priv];
      } else {
        bufA = [device newBufferWithLength:bytes_fp16 options:shared];
        bufB = [device newBufferWithLength:bytes_fp16 options:shared];
        bufC = [device newBufferWithLength:bytes_fp32 options:shared];
      }
      if (!bufA || !bufB || !bufC || (use_private && (!stageA || !stageB))) {
        error = "failed to allocate MTLBuffers (fp16)";
        return false;
      }
      __fp16* a = reinterpret_cast<__fp16*>((use_private ? stageA : bufA).contents);
      __fp16* b = reinterpret_cast<__fp16*>((use_private ? stageB : bufB).contents);
      fill_fp16(a, n, 1);
      fill_fp16(b, n, 2);
    } else if (opt.precision == GpuPrecision::BF16) {
      if (use_private) {
        stageA = [device newBufferWithLength:bytes_bf16 options:shared];
        stageB = [device newBufferWithLength:bytes_bf16 options:shared];
        bufA = [device newBufferWithLength:bytes_bf16 options:priv];
        bufB = [device newBufferWithLength:bytes_bf16 options:priv];
        bufC = [device newBufferWithLength:bytes_fp32 options:priv];
      } else {
        bufA = [device newBufferWithLength:bytes_bf16 options:shared];
        bufB = [device newBufferWithLength:bytes_bf16 options:shared];
        bufC = [device newBufferWithLength:bytes_fp32 options:shared];
      }
      if (!bufA || !bufB || !bufC || (use_private && (!stageA || !stageB))) {
        error = "failed to allocate MTLBuffers (bf16)";
        return false;
      }
      uint16_t* a = reinterpret_cast<uint16_t*>((use_private ? stageA : bufA).contents);
      uint16_t* b = reinterpret_cast<uint16_t*>((use_private ? stageB : bufB).contents);
      fill_bf16(a, n, 1);
      fill_bf16(b, n, 2);
    } else {  // INT8
      if (use_private) {
        stageA = [device newBufferWithLength:bytes_int8 options:shared];
        stageB = [device newBufferWithLength:bytes_int8 options:shared];
        bufA = [device newBufferWithLength:bytes_int8 options:priv];
        bufB = [device newBufferWithLength:bytes_int8 options:priv];
        bufC = [device newBufferWithLength:bytes_fp32 options:priv];
      } else {
        bufA = [device newBufferWithLength:bytes_int8 options:shared];
        bufB = [device newBufferWithLength:bytes_int8 options:shared];
        bufC = [device newBufferWithLength:bytes_fp32 options:shared];
      }
      if (!bufA || !bufB || !bufC || (use_private && (!stageA || !stageB))) {
        error = "failed to allocate MTLBuffers (int8)";
        return false;
      }
      int8_t* a = reinterpret_cast<int8_t*>((use_private ? stageA : bufA).contents);
      int8_t* b = reinterpret_cast<int8_t*>((use_private ? stageB : bufB).contents);
      fill_int8(a, n, 1);
      fill_int8(b, n, 2);
    }

    const uint32_t N_u32 = static_cast<uint32_t>(n);
    const uint32_t inner_u32 = static_cast<uint32_t>(opt.inner);

    auto build_pso = [&](GpuKernelVariant k) -> id<MTLComputePipelineState> {
      if (auto pso = cached_pso(opt.precision, k)) return pso;
      const char* name = kernel_name(opt.precision, k);
      NSString* fnName = [NSString stringWithUTF8String:name];
      id<MTLFunction> fn = [lib newFunctionWithName:fnName];
      if (!fn) return nil;
      id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&nsErr];
      if (pso) set_cached_pso(opt.precision, k, pso);
      return pso;
    };

    auto build_peak_pso = [&]() -> id<MTLComputePipelineState> {
      if (auto pso = cached_peak_pso(opt.precision)) return pso;
      const char* name = (opt.precision == GpuPrecision::FP32) ? "peak_fp32"
                          : (opt.precision == GpuPrecision::FP16) ? "peak_fp16"
                          : (opt.precision == GpuPrecision::BF16) ? "peak_bf16"
                                                                  : "peak_int8";
      NSString* fnName = [NSString stringWithUTF8String:name];
      id<MTLFunction> fn = [lib newFunctionWithName:fnName];
      if (!fn) return nil;
      id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&nsErr];
      if (pso) set_cached_peak_pso(opt.precision, pso);
      return pso;
    };

    auto encode_once_gemm = [&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso,
                                GpuKernelVariant k) {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:bufA offset:0 atIndex:0];
      [enc setBuffer:bufB offset:0 atIndex:1];
      [enc setBuffer:bufC offset:0 atIndex:2];
      [enc setBytes:&N_u32 length:sizeof(N_u32) atIndex:3];
      [enc setBytes:&inner_u32 length:sizeof(inner_u32) atIndex:4];

      (void)k;
      const bool int8 = (opt.precision == GpuPrecision::INT8);
      const int tile = int8 ? 16 : 32;
      const MTLSize tg = MTLSizeMake(static_cast<NSUInteger>((n + tile - 1) / tile),
                                     static_cast<NSUInteger>((n + tile - 1) / tile),
                                     1);
      const MTLSize th = int8 ? MTLSizeMake(16, 16, 1) : MTLSizeMake(128, 1, 1);
      for (int i = 0; i < opt.batch; i++) {
        [enc dispatchThreadgroups:tg threadsPerThreadgroup:th];
      }
      [enc endEncoding];
    };

    auto encode_once_peak = [&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso,
                                id<MTLBuffer> outBuf) {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:outBuf offset:0 atIndex:0];
      [enc setBytes:&inner_u32 length:sizeof(inner_u32) atIndex:1];

      const int tile = 8;
      const MTLSize tg = MTLSizeMake(static_cast<NSUInteger>(n / tile),
                                     static_cast<NSUInteger>(n / tile),
                                     1);
      const bool int8 = (opt.precision == GpuPrecision::INT8);
      const MTLSize th = int8 ? MTLSizeMake(32, 1, 1) : MTLSizeMake(128, 1, 1);
      for (int i = 0; i < opt.batch; i++) {
        [enc dispatchThreadgroups:tg threadsPerThreadgroup:th];
      }
      [enc endEncoding];
    };

    std::vector<GpuKernelVariant> kernels;
    if (opt.kernel == GpuKernelVariant::Auto) {
      kernels.push_back(GpuKernelVariant::V4);
    } else {
      kernels.push_back(opt.kernel);
    }

    if (opt.workload == GpuWorkload::Gemm && use_private) {
      // 把 A/B staging 拷贝到 private buffer（单独 command buffer，避免计入测量）。
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      const size_t bytes_a = input_bytes();
      const size_t bytes_b = input_bytes();
      [blit copyFromBuffer:stageA sourceOffset:0 toBuffer:bufA destinationOffset:0 size:bytes_a];
      [blit copyFromBuffer:stageB sourceOffset:0 toBuffer:bufB destinationOffset:0 size:bytes_b];
      [blit endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) {
        error = "blit command buffer failed";
        return false;
      }
    }

    if (opt.workload == GpuWorkload::Peak) {
      id<MTLComputePipelineState> pso = build_peak_pso();
      if (!pso) {
        error = "failed to create compute pipeline for peak (" +
                std::string([[ns_error(nsErr) description] UTF8String]) + ")";
        return false;
      }
      const NSUInteger needed_threads = (opt.precision == GpuPrecision::INT8) ? 32 : 128;
      if (pso.maxTotalThreadsPerThreadgroup < needed_threads) {
        error = "peak pipeline maxTotalThreadsPerThreadgroup too small";
        return false;
      }

      const int tg_dim = n / 8;
      const size_t out_bytes = static_cast<size_t>(tg_dim) * static_cast<size_t>(tg_dim) * sizeof(float);
      const MTLResourceOptions shared = MTLResourceStorageModeShared;
      const MTLResourceOptions priv = MTLResourceStorageModePrivate;
      id<MTLBuffer> outBuf = [device newBufferWithLength:out_bytes options:(use_private ? priv : shared)];
      if (!outBuf) {
        error = "failed to allocate peak output buffer";
        return false;
      }

      auto measure_peak = [&](int warmup, int repeats, double& best_out) -> bool {
        for (int i = 0; i < warmup; i++) {
          id<MTLCommandBuffer> cb = [queue commandBuffer];
          encode_once_peak(cb, pso, outBuf);
          [cb commit];
          [cb waitUntilCompleted];
          if (cb.status != MTLCommandBufferStatusCompleted) {
            error = "warmup command buffer failed";
            return false;
          }
        }

        double best = std::numeric_limits<double>::infinity();
        for (int i = 0; i < repeats; i++) {
          id<MTLCommandBuffer> cb = [queue commandBuffer];
          encode_once_peak(cb, pso, outBuf);
          [cb commit];
          [cb waitUntilCompleted];
          if (cb.status != MTLCommandBufferStatusCompleted) {
            NSError* e = cb.error;
            error = "command buffer failed: " + std::string([[ns_error(e) description] UTF8String]);
            return false;
          }
          const double dt = cb.GPUEndTime - cb.GPUStartTime;
          if (!(dt > 0.0) || !std::isfinite(dt)) {
            error = "GPUStartTime/GPUEndTime unavailable (dt <= 0).";
            return false;
          }
          best = std::min(best, dt);
        }
        best_out = best;
        return true;
      };

      double best = 0.0;
      if (!measure_peak(opt.warmup, opt.repeats, best)) return false;
      out.best_seconds = best;
      out.score = score_for_peak(opt.precision, n, opt.batch, opt.inner, best);
      out.used_kernel = GpuKernelVariant::Auto;
      return true;
    }

    auto measure = [&](GpuKernelVariant k, int warmup, int repeats, double& best_out) -> bool {
      id<MTLComputePipelineState> pso = build_pso(k);
      if (!pso) {
        if (opt.precision == GpuPrecision::BF16) {
          error = "bf16 not supported by current Metal compiler/runtime (missing bfloat/simdgroup support)";
          return false;
        }
        error = "failed to create compute pipeline for: " + std::string(kernel_name(opt.precision, k)) +
                " (" + std::string([[ns_error(nsErr) description] UTF8String]) + ")";
        return false;
      }

      (void)k;
      const NSUInteger needed_threads = (opt.precision == GpuPrecision::INT8) ? 256 : 128;
      if (pso.maxTotalThreadsPerThreadgroup < needed_threads) {
        error = "pipeline maxTotalThreadsPerThreadgroup too small for selected kernel";
        return false;
      }

      for (int i = 0; i < warmup; i++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        encode_once_gemm(cb, pso, k);
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
          error = "warmup command buffer failed";
          return false;
        }
      }

      double best = std::numeric_limits<double>::infinity();
      for (int i = 0; i < repeats; i++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        encode_once_gemm(cb, pso, k);
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status != MTLCommandBufferStatusCompleted) {
          NSError* e = cb.error;
          error = "command buffer failed: " + std::string([[ns_error(e) description] UTF8String]);
          return false;
        }

        const double dt = cb.GPUEndTime - cb.GPUStartTime;
        if (!(dt > 0.0) || !std::isfinite(dt)) {
          error = "GPUStartTime/GPUEndTime unavailable (dt <= 0). Ensure macOS supports GPU timestamps.";
          return false;
        }
        best = std::min(best, dt);
      }

      best_out = best;
      return true;
    };

    GpuKernelVariant chosen = kernels.front();
    // 发布版仅保留 v4，auto 等同 v4。

    double best = 0.0;
    if (!measure(chosen, opt.warmup, opt.repeats, best)) return false;

    out.best_seconds = best;
    out.score = score_for_gemm(n, opt.batch, opt.inner, best);
    out.used_kernel = chosen;

    if (opt.readback_c) {
      if (opt.workload != GpuWorkload::Gemm) {
        error = "readback_c only supported for GEMM workload";
        return false;
      }
      if (use_private) {
        error = "readback_c requires gpu-storage=shared (private buffers have no CPU-visible contents)";
        return false;
      }
      float* c_ptr = reinterpret_cast<float*>(bufC.contents);
      if (!c_ptr) {
        error = "failed to map C buffer contents";
        return false;
      }
      const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
      opt.readback_c->assign(c_ptr, c_ptr + nn);
    }
    return true;
  }
}
