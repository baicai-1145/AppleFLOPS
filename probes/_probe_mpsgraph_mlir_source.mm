#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

@interface MPSGraphExecutableDescriptor : NSObject
@end

@interface MPSGraphExecutable (PrivateMLIRSource)
- (instancetype)initWithMLIRSourceFromURL:(NSURL*)url executableDescriptor:(MPSGraphExecutableDescriptor*)descriptor;
@end

static void fill_i8(std::vector<int8_t>& v, uint32_t seed) {
  uint32_t x = seed;
  for (size_t i = 0; i < v.size(); ++i) {
    x = x * 1664525u + 1013904223u;
    v[i] = static_cast<int8_t>(((x >> 24) & 0x0f) - 8);
  }
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) {
      std::fprintf(stderr, "usage: %s module.mlir n [repeats]\n", argv[0]);
      return 2;
    }
    NSString* path = [NSString stringWithUTF8String:argv[1]];
    const int n = std::atoi(argv[2]);
    const int repeats = (argc > 3) ? std::atoi(argv[3]) : 3;
    if (n <= 0 || repeats <= 0) return 2;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!device || !queue) {
      std::fprintf(stderr, "failed to create Metal device/queue\n");
      return 1;
    }
    std::printf("device=%s supportsApple9=%d supportsApple10=%d mlir=%s n=%d repeats=%d\n",
                device.name.UTF8String,
                [device supportsFamily:MTLGPUFamilyApple9],
                [device supportsFamily:MTLGPUFamilyApple10],
                path.UTF8String, n, repeats);

    MPSGraphExecutableDescriptor* desc = [NSClassFromString(@"MPSGraphExecutableDescriptor") new];
    NSURL* url = [NSURL fileURLWithPath:path];
    MPSGraphExecutable* executable = nil;
    @try {
      executable = [[MPSGraphExecutable alloc] initWithMLIRSourceFromURL:url executableDescriptor:desc];
    } @catch (NSException* e) {
      std::fprintf(stderr, "init exception: %s: %s\n", e.name.UTF8String, e.reason.UTF8String);
      return 1;
    }
    if (!executable) {
      std::fprintf(stderr, "initWithMLIRSourceFromURL returned nil\n");
      return 1;
    }

    if (std::getenv("MPSGRAPH_GET_IR")) {
      SEL getIR = NSSelectorFromString(@"getIR");
      if ([executable respondsToSelector:getIR]) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Warc-performSelector-leaks"
        id ir = [executable performSelector:getIR];
#pragma clang diagnostic pop
        NSString* text = [ir description];
        const char* outPath = std::getenv("MPSGRAPH_GET_IR_PATH");
        if (outPath && *outPath) {
          NSError* writeError = nil;
          BOOL ok = [text writeToFile:[NSString stringWithUTF8String:outPath]
                            atomically:YES
                              encoding:NSUTF8StringEncoding
                                 error:&writeError];
          std::printf("getIR class=%s write=%s path=%s%s%s\n",
                      ir ? object_getClassName(ir) : "(nil)",
                      ok ? "ok" : "fail",
                      outPath,
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

    MPSShape* shape = @[@(n), @(n)];
    MPSGraphDevice* graphDevice = [MPSGraphDevice deviceWithMTLDevice:device];
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

    @try {
      (void)[executable runWithMTLCommandQueue:queue
                                   inputsArray:@[aTensorData, bTensorData]
                                  resultsArray:nil
                           executionDescriptor:nil];
    } @catch (NSException* e) {
      std::fprintf(stderr, "warmup exception: %s: %s\n", e.name.UTF8String, e.reason.UTF8String);
      return 1;
    }

    double best_ms = 1e100;
    for (int i = 0; i < repeats; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      @try {
        (void)[executable runWithMTLCommandQueue:queue
                                     inputsArray:@[aTensorData, bTensorData]
                                    resultsArray:nil
                             executionDescriptor:nil];
      } @catch (NSException* e) {
        std::fprintf(stderr, "run exception: %s: %s\n", e.name.UTF8String, e.reason.UTF8String);
        return 1;
      }
      const auto t1 = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (ms < best_ms) best_ms = ms;
    }
    std::printf("best_ms=%.3f\n", best_ms);
    return 0;
  }
}
