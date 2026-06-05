#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void fill_i8(std::vector<int8_t>& v, uint32_t seed) {
  uint32_t x = seed;
  for (size_t i = 0; i < v.size(); ++i) {
    x = x * 1664525u + 1013904223u;
    v[i] = static_cast<int8_t>(((x >> 24) & 0x0f) - 8);
  }
}

static const char* mode_help() {
  return "mode must be one of: dequant-f16, dequant-f32, direct-i8";
}

int main(int argc, char** argv) {
  @autoreleasepool {
    const std::string mode = (argc > 1) ? argv[1] : "dequant-f16";
    const int n = (argc > 2) ? std::atoi(argv[2]) : 512;
    const int repeats = (argc > 3) ? std::atoi(argv[3]) : 5;
    if (n <= 0 || repeats <= 0) {
      std::fprintf(stderr, "usage: %s [dequant-f16|dequant-f32|direct-i8] [n] [repeats]\n", argv[0]);
      return 2;
    }
    if (mode != "dequant-f16" && mode != "dequant-f32" && mode != "direct-i8") {
      std::fprintf(stderr, "%s\n", mode_help());
      return 2;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::fprintf(stderr, "no Metal device\n");
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      std::fprintf(stderr, "failed to create command queue\n");
      return 1;
    }

    const bool apple9 = [device supportsFamily:MTLGPUFamilyApple9];
    const bool apple10 = [device supportsFamily:MTLGPUFamilyApple10];
    std::printf("device=%s supportsApple9=%d supportsApple10=%d mode=%s n=%d repeats=%d\n",
                device.name.UTF8String, apple9, apple10, mode.c_str(), n, repeats);

    MPSGraphDevice* graphDevice = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSShape* shape = @[@(n), @(n)];
    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);

    std::vector<int8_t> aHost(elems), bHost(elems);
    fill_i8(aHost, 1);
    fill_i8(bHost, 2);
    NSData* aData = [NSData dataWithBytes:aHost.data() length:aHost.size() * sizeof(int8_t)];
    NSData* bData = [NSData dataWithBytes:bHost.data() length:bHost.size() * sizeof(int8_t)];
    MPSGraphTensorData* aTensorData =
        [[MPSGraphTensorData alloc] initWithDevice:graphDevice data:aData shape:shape dataType:MPSDataTypeInt8];
    MPSGraphTensorData* bTensorData =
        [[MPSGraphTensorData alloc] initWithDevice:graphDevice data:bData shape:shape dataType:MPSDataTypeInt8];
    if (!aTensorData || !bTensorData) {
      std::fprintf(stderr, "failed to create MPSGraphTensorData\n");
      return 1;
    }

    MPSGraph* graph = [MPSGraph new];
    graph.options = static_cast<MPSGraphOptions>(MPSGraphOptionsSynchronizeResults | MPSGraphOptionsVerbose);
    MPSGraphTensor* a = [graph placeholderWithShape:shape dataType:MPSDataTypeInt8 name:@"A_i8"];
    MPSGraphTensor* b = [graph placeholderWithShape:shape dataType:MPSDataTypeInt8 name:@"B_i8"];
    MPSGraphTensor* c = nil;

    if (mode == "direct-i8") {
      c = [graph matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:@"matmul_direct_i8"];
    } else {
      const MPSDataType deqType = (mode == "dequant-f16") ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
      MPSGraphTensor* af = [graph dequantizeTensor:a scale:1.0 zeroPoint:0.0 dataType:deqType name:@"A_deq"];
      MPSGraphTensor* bf = [graph dequantizeTensor:b scale:1.0 zeroPoint:0.0 dataType:deqType name:@"B_deq"];
      c = [graph matrixMultiplicationWithPrimaryTensor:af secondaryTensor:bf name:@"matmul_dequant"];
    }

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{a : aTensorData, b : bTensorData};
    NSArray<MPSGraphTensor*>* targets = @[c];

    if (std::getenv("MPSGRAPH_GET_IR")) {
      MPSGraphShapedType* i8Shape = [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeInt8];
      NSDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feedTypes = @{a : i8Shape, b : i8Shape};
      MPSGraphCompilationDescriptor* cd = [MPSGraphCompilationDescriptor new];
      cd.waitForCompilationCompletion = YES;
      cd.optimizationLevel = MPSGraphOptimizationLevel1;
      MPSGraphExecutable* executable = nil;
      @try {
        executable = [graph compileWithDevice:graphDevice
                                        feeds:feedTypes
                                targetTensors:targets
                             targetOperations:nil
                        compilationDescriptor:cd];
      } @catch (NSException* e) {
        std::fprintf(stderr, "compile exception: %s: %s\n", e.name.UTF8String, e.reason.UTF8String);
        return 1;
      }

      SEL getIR = NSSelectorFromString(@"getIR");
      if (executable && [executable respondsToSelector:getIR]) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Warc-performSelector-leaks"
        id ir = [executable performSelector:getIR];
#pragma clang diagnostic pop
        NSString* text = [ir description];
        const char* path = std::getenv("MPSGRAPH_GET_IR_PATH");
        if (path && *path) {
          NSError* writeError = nil;
          BOOL ok = [text writeToFile:[NSString stringWithUTF8String:path]
                            atomically:YES
                              encoding:NSUTF8StringEncoding
                                 error:&writeError];
          std::printf("getIR class=%s write=%s path=%s%s%s\n",
                      ir ? object_getClassName(ir) : "(nil)",
                      ok ? "ok" : "fail",
                      path,
                      writeError ? " error=" : "",
                      writeError ? writeError.localizedDescription.UTF8String : "");
        } else {
          std::printf("getIR class=%s\n%s\n", ir ? object_getClassName(ir) : "(nil)",
                      text ? text.UTF8String : "(nil)");
        }
      } else {
        std::printf("getIR unavailable\n");
      }
    }

    @try {
      (void)[graph runWithMTLCommandQueue:queue feeds:feeds targetTensors:targets targetOperations:nil];
    } @catch (NSException* e) {
      std::fprintf(stderr, "warmup exception: %s: %s\n", e.name.UTF8String, e.reason.UTF8String);
      return 1;
    }

    double best_ms = 1e100;
    double total_ms = 0.0;
    for (int r = 0; r < repeats; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      @try {
        (void)[graph runWithMTLCommandQueue:queue feeds:feeds targetTensors:targets targetOperations:nil];
      } @catch (NSException* e) {
        std::fprintf(stderr, "run exception: %s: %s\n", e.name.UTF8String, e.reason.UTF8String);
        return 1;
      }
      const auto t1 = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (ms < best_ms) best_ms = ms;
      total_ms += ms;
    }

    const double ops = 2.0 * n * n * n;
    const double tops_best = ops / (best_ms / 1000.0) / 1e12;
    std::printf("avg_ms=%.3f best_ms=%.3f nominal_ops=%.3e nominal_TOPS=%.3f\n",
                total_ms / repeats, best_ms, ops, tops_best);
    return 0;
  }
}
