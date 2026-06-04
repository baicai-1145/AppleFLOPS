#pragma once

#include <string>
#include <vector>

enum class GpuPrecision {
  FP32,
  FP16,
  BF16,
  INT8,
};

enum class GpuKernelVariant {
  Auto,  // 运行时默认选择 v4（保留该选项用于 CLI 兼容）
  V4,    // 32x32 tile + TK=32 + padding + vector loads（更高算术强度/更少 barrier）
};

enum class GpuStorageMode {
  Shared,   // MTLResourceStorageModeShared
  Private,  // MTLResourceStorageModePrivate（推荐，通常更快）
};

enum class GpuWorkload {
  Gemm,  // 真实 GEMM（仍受 A/B 访存与 C 写回影响）
  Peak,  // compute-amplified（极小写回，尽量逼近理论峰值）
};

struct GpuBenchOptions {
  int n = 1024;
  int warmup = 1;
  int repeats = 5;
  GpuPrecision precision = GpuPrecision::FP16;
  std::string shader_path = "shaders/gemm.metal";
  GpuKernelVariant kernel = GpuKernelVariant::Auto;
  GpuStorageMode storage = GpuStorageMode::Private;
  int batch = 1;  // 每个 command buffer 内重复 dispatch 次数（用于拉长 GPU 时间，提升稳定性）
  int inner = 1;  // kernel 内重复同一 tile 的 MMA 次数（compute-amplified，接近峰值用）
  GpuWorkload workload = GpuWorkload::Gemm;

  // 可选：把最后一次执行的 C 读回到 CPU（用于 correctness verify）。
  // 为保持 benchmark 纯净，这里只支持 Shared storage；请在调用方显式设置 storage=Shared。
  std::vector<float>* readback_c = nullptr;  // size=n*n
};

struct GpuBenchResult {
  double best_seconds = 0.0;  // GPUStart/GPUEndTime 计算得到的 GPU 硬件时间
  double score = 0.0;  // FP/BF16: TFLOPS, INT8: TOPS
  GpuKernelVariant used_kernel = GpuKernelVariant::Auto;
  std::string backend;
};

// 返回 true 表示运行成功；失败时 error 会带原因（例如 shader 编译失败、设备不支持等）。
bool run_gpu_bench(const GpuBenchOptions& opt, GpuBenchResult& out, std::string& error);
