#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

struct Config {
  int n = 1024;
  int repeats = 1;
  bool rank3 = false;
  bool transposeB = false;
  bool scaleScalar = false;
  int scaleRank = 2;
  MPSDataType scaleType = MPSDataTypeFloat32;
  MPSDataType outputType = MPSDataTypeInt32;
};

static void fill_int8(int8_t* p, size_t count, uint32_t seed) {
  uint32_t x = seed ? seed : 1u;
  for (size_t i = 0; i < count; ++i) {
    x = x * 1664525u + 1013904223u;
    p[i] = static_cast<int8_t>(((x >> 24) & 0x0f) - 8);
  }
}

static const char* dtype_name(MPSDataType t) {
  switch (t) {
    case MPSDataTypeFloat16: return "f16";
    case MPSDataTypeFloat32: return "f32";
    case MPSDataTypeInt32: return "i32";
    case MPSDataTypeInt8: return "i8";
    default: return "other";
  }
}

static bool parse_variant(const char* s, Config& cfg) {
  std::string v = s ? s : "";
  auto has = [&](const char* token) { return v.find(token) != std::string::npos; };
  if (has("rank3")) cfg.rank3 = true;
  if (has("nt")) cfg.transposeB = true;
  if (has("scalar")) cfg.scaleScalar = true;
  if (has("sr1")) cfg.scaleRank = 1;
  if (has("sr2")) cfg.scaleRank = 2;
  if (has("sr3")) cfg.scaleRank = 3;
  if (has("sf16")) cfg.scaleType = MPSDataTypeFloat16;
  if (has("sf32")) cfg.scaleType = MPSDataTypeFloat32;
  if (has("of16")) cfg.outputType = MPSDataTypeFloat16;
  if (has("of32")) cfg.outputType = MPSDataTypeFloat32;
  if (has("oi32")) cfg.outputType = MPSDataTypeInt32;
  return !v.empty();
}

static void print_array(const char* name, MPSNDArray* a) {
  printf("%s dtype=0x%x/%s rank=%lu shape=[", name, a.dataType, dtype_name(a.dataType),
         static_cast<unsigned long>(a.numberOfDimensions));
  for (NSUInteger i = 0; i < a.numberOfDimensions; ++i) {
    if (i) printf(",");
    printf("%lu", static_cast<unsigned long>([a lengthOfDimension:i]));
  }
  printf("]\n");
}

static MPSNDArrayDescriptor* make_desc(MPSDataType type, int rank, int n) {
  NSUInteger dims[3] = {static_cast<NSUInteger>(n), static_cast<NSUInteger>(n), 1};
  MPSNDArrayDescriptor* desc =
      [MPSNDArrayDescriptor descriptorWithDataType:type
                                    dimensionCount:static_cast<NSUInteger>(rank)
                                    dimensionSizes:dims];
  desc.preferPackedRows = YES;
  return desc;
}

static MPSNDArray* make_scale(id<MTLDevice> device, const Config& cfg) {
  if (cfg.scaleScalar) {
    return [[MPSNDArray alloc] initWithDevice:device scalar:1.0];
  }

  NSUInteger dims[3] = {1, 1, 1};
  MPSNDArrayDescriptor* desc =
      [MPSNDArrayDescriptor descriptorWithDataType:cfg.scaleType
                                    dimensionCount:static_cast<NSUInteger>(cfg.scaleRank)
                                    dimensionSizes:dims];
  desc.preferPackedRows = YES;
  MPSNDArray* scale = [[MPSNDArray alloc] initWithDevice:device descriptor:desc];
  if (!scale) return nil;
  if (cfg.scaleType == MPSDataTypeFloat16) {
    __fp16 one = static_cast<__fp16>(1.0f);
    [scale writeBytes:&one strideBytes:nil];
  } else {
    float one = 1.0f;
    [scale writeBytes:&one strideBytes:nil];
  }
  return scale;
}

static NSString* ns_error(NSError* e) {
  if (!e) return @"(nil)";
  return [NSString stringWithFormat:@"%@ (code=%ld)", e.localizedDescription, (long)e.code];
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    Config cfg;
    if (argc > 1 && !parse_variant(argv[1], cfg)) {
      fprintf(stderr, "bad variant\n");
      return 2;
    }
    if (argc > 2) cfg.n = std::atoi(argv[2]);
    if (argc > 3) cfg.repeats = std::atoi(argv[3]);
    if (cfg.n <= 0 || cfg.repeats <= 0) return 2;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!device || !queue) {
      fprintf(stderr, "failed to create Metal device/queue\n");
      return 1;
    }

    printf("device=%s n=%d repeats=%d rank=%d B=%s scale=%s/%s rank=%d out=%s\n",
           device.name.UTF8String, cfg.n, cfg.repeats, cfg.rank3 ? 3 : 2,
           cfg.transposeB ? "T" : "N", cfg.scaleScalar ? "scalar" : "array",
           dtype_name(cfg.scaleType), cfg.scaleRank, dtype_name(cfg.outputType));

    const int rank = cfg.rank3 ? 3 : 2;
    MPSNDArrayDescriptor* inputDesc = make_desc(MPSDataTypeInt8, rank, cfg.n);
    MPSNDArrayDescriptor* outputDesc = make_desc(cfg.outputType, rank, cfg.n);
    MPSNDArray* A = [[MPSNDArray alloc] initWithDevice:device descriptor:inputDesc];
    MPSNDArray* BStorage = [[MPSNDArray alloc] initWithDevice:device descriptor:inputDesc];
    MPSNDArray* C = [[MPSNDArray alloc] initWithDevice:device descriptor:outputDesc];
    MPSNDArray* scaleA = make_scale(device, cfg);
    MPSNDArray* scaleBStorage = make_scale(device, cfg);
    if (!A || !BStorage || !C || !scaleA || !scaleBStorage) {
      fprintf(stderr, "failed to allocate NDArrays\n");
      return 1;
    }

    const size_t count = static_cast<size_t>(cfg.n) * static_cast<size_t>(cfg.n);
    std::vector<int8_t> a(count), b(count);
    fill_int8(a.data(), count, 1);
    fill_int8(b.data(), count, 2);
    [A writeBytes:a.data() strideBytes:nil];
    [BStorage writeBytes:b.data() strideBytes:nil];

    MPSNDArray* B = BStorage;
    MPSNDArray* scaleB = scaleBStorage;
    if (cfg.transposeB) {
      MPSNDArrayDescriptor* bViewDesc = [BStorage descriptor];
      [bViewDesc transposeDimension:0 withDimension:1];
      B = [BStorage arrayViewWithDescriptor:bViewDesc];

      if (!cfg.scaleScalar) {
        MPSNDArrayDescriptor* scaleBViewDesc = [scaleBStorage descriptor];
        if (scaleBViewDesc.numberOfDimensions >= 2) {
          [scaleBViewDesc transposeDimension:0 withDimension:1];
        }
        scaleB = [scaleBStorage arrayViewWithDescriptor:scaleBViewDesc];
      }
    }
    if (!B || !scaleB) {
      fprintf(stderr, "failed to create transpose view\n");
      return 1;
    }

    print_array("A", A);
    print_array("B", B);
    print_array("C", C);
    print_array("scaleA", scaleA);
    print_array("scaleB", scaleB);

    MPSNDArrayAffineQuantizationDescriptor* qA =
        [[MPSNDArrayAffineQuantizationDescriptor alloc] initWithDataType:MPSDataTypeInt8
                                                            hasZeroPoint:NO
                                                             hasMinValue:NO];
    MPSNDArrayAffineQuantizationDescriptor* qB =
        [[MPSNDArrayAffineQuantizationDescriptor alloc] initWithDataType:MPSDataTypeInt8
                                                            hasZeroPoint:NO
                                                             hasMinValue:NO];
    MPSNDArrayQuantizedMatrixMultiplication* qmm =
        [[MPSNDArrayQuantizedMatrixMultiplication alloc] initWithDevice:device
                                             leftQuantizationDescriptor:qA
                                            rightQuantizationDescriptor:qB];
    qmm.beta = 0.0;
    if (!qmm) {
      fprintf(stderr, "failed to create QMM\n");
      return 1;
    }

    NSArray<MPSNDArray*>* sources = @[ A, B, scaleA, scaleB ];
    id<MTLCommandBuffer> warmup = [queue commandBuffer];
    [qmm encodeToCommandBuffer:warmup sourceArrays:sources destinationArray:C];
    [warmup commit];
    [warmup waitUntilCompleted];
    if (warmup.status != MTLCommandBufferStatusCompleted) {
      fprintf(stderr, "warmup failed: %s\n", ns_error(warmup.error).UTF8String);
      return 1;
    }

    for (int i = 0; i < cfg.repeats; ++i) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      [qmm encodeToCommandBuffer:cb sourceArrays:sources destinationArray:C];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "run failed: %s\n", ns_error(cb.error).UTF8String);
        return 1;
      }
      printf("run=%d gpu_time=%.9f\n", i, cb.GPUEndTime - cb.GPUStartTime);
    }

    if (const char* sleepMs = std::getenv("PROBE_SLEEP_MS")) {
      const int ms = std::atoi(sleepMs);
      if (ms > 0) usleep(static_cast<useconds_t>(ms) * 1000);
    }
  }
  return 0;
}
