#pragma once

#include <string>

enum class NpuPrecision {
  FP32,
  FP16,
  BF16,
  INT8,
};

struct NpuBenchOptions {
  int channels = 512;
  int spatial = 64;
  int depth = 64;
  int warmup = 5;
  int repeats = 10;
  NpuPrecision precision = NpuPrecision::FP16;
};

struct NpuBenchResult {
  double best_seconds = 0.0;
  double score = 0.0;  // FP/BF16/FP32: TFLOPS, INT8: TOPS
  double operations = 0.0;
  bool npu_only = false;
  std::string backend;
  std::string note;
};

bool run_npu_bench(const NpuBenchOptions& opt, NpuBenchResult& out, std::string& error);
