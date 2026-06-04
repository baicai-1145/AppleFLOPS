#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include "npu_bench.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mach/mach_time.h>
#include <cstdio>

namespace {

mach_timebase_info_data_t& timebase() {
  static mach_timebase_info_data_t tb = [] {
    mach_timebase_info_data_t v;
    mach_timebase_info(&v);
    return v;
  }();
  return tb;
}

double ticks_to_seconds(uint64_t t) {
  const auto& tb = timebase();
  return static_cast<double>(t) * static_cast<double>(tb.numer) /
         static_cast<double>(tb.denom) / 1e9;
}

const char* precision_name(NpuPrecision p) {
  if (p == NpuPrecision::FP32) return "fp32";
  if (p == NpuPrecision::FP16) return "fp16";
  if (p == NpuPrecision::BF16) return "bf16";
  return "int8";
}

const char* graph_kind_name(NpuPrecision p) {
  if (p == NpuPrecision::FP32) return "fp32_io_fp16_compute";
  if (p == NpuPrecision::BF16) return "bf16_io_fp16_compute";
  if (p == NpuPrecision::INT8) return "w8a8_quantized_chain";
  return "fp16_conv_chain";
}

int element_bytes(NpuPrecision p) {
  if (p == NpuPrecision::FP32) return 4;
  if (p == NpuPrecision::INT8) return 2;  // W8A8 graph still uses fp16 external IO.
  return 2;
}

uint16_t float_to_bf16_bits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t lsb = (u >> 16) & 1u;
  const uint32_t bias = 0x7FFFu + lsb;
  return static_cast<uint16_t>((u + bias) >> 16);
}

NSData* build_weight_blob(NpuPrecision p, int ch, int depth) {
  const int bytes = (p == NpuPrecision::FP32) ? 4 : (p == NpuPrecision::INT8) ? 1 : 2;
  const NSUInteger wsize = static_cast<NSUInteger>(ch) * static_cast<NSUInteger>(ch) * bytes;
  const NSUInteger chunk_size = 64 + wsize;
  const NSUInteger total = 64 + chunk_size * static_cast<NSUInteger>(depth);
  uint8_t* buf = static_cast<uint8_t*>(std::calloc(total, 1));
  if (!buf) return nil;

  buf[0] = 0x01;
  buf[4] = 0x02;
  uint32_t x = 0x12345678u;
  for (int i = 0; i < depth; i++) {
    uint8_t* chunk = buf + 64 + static_cast<NSUInteger>(i) * chunk_size;
    chunk[0] = 0xEF;
    chunk[1] = 0xBE;
    chunk[2] = 0xAD;
    chunk[3] = 0xDE;
    chunk[4] = 0x01;
    chunk[10] = (p == NpuPrecision::INT8) ? 0x08 : (p == NpuPrecision::FP32) ? 0x20 : 0x10;

    if (p == NpuPrecision::INT8) {
      int8_t* data = reinterpret_cast<int8_t*>(chunk + 64);
      for (NSUInteger j = 0; j < wsize; j++) {
        x = x * 1664525u + 1013904223u;
        data[j] = static_cast<int8_t>((x >> 24) - 128);
      }
    } else if (p == NpuPrecision::FP32) {
      float* data = reinterpret_cast<float*>(chunk + 64);
      const NSUInteger count = static_cast<NSUInteger>(ch) * static_cast<NSUInteger>(ch);
      for (NSUInteger j = 0; j < count; j++) {
        x = x * 1664525u + 1013904223u;
        data[j] = (static_cast<float>((x >> 8) & 0xFFFFu) / 65536.0f - 0.5f) * 0.01f;
      }
    } else {
      uint16_t* data = reinterpret_cast<uint16_t*>(chunk + 64);
      const NSUInteger count = static_cast<NSUInteger>(ch) * static_cast<NSUInteger>(ch);
      for (NSUInteger j = 0; j < count; j++) {
        x = x * 1664525u + 1013904223u;
        const float f = (static_cast<float>((x >> 8) & 0xFFFFu) / 65536.0f - 0.5f) * 0.01f;
        if (p == NpuPrecision::BF16) {
          data[j] = float_to_bf16_bits(f);
        } else {
          reinterpret_cast<_Float16*>(data)[j] = static_cast<_Float16>(f);
        }
      }
    }
  }

  return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

NSString* mil_header() {
  return @"program(1.3)\n"
          "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
          "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
          "{\"coremltools-version\", \"9.0\"}})]\n"
          "{\n";
}

void append_conv_consts(NSMutableString* m) {
  [m appendString:@"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
                  @"        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
                  @"        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
                  @"        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
                  @"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"];
}

NSString* gen_mil_typed(NpuPrecision p, int ch, int sp, int depth) {
  const char* dtype = precision_name(p);
  NSMutableString* m = [NSMutableString stringWithString:mil_header()];
  [m appendFormat:@"    func main<ios18>(tensor<%s, [1, %d, %d, %d]> x) {\n", dtype, ch, sp, sp];
  append_conv_consts(m);

  const NSUInteger bytes = (p == NpuPrecision::FP32) ? 4 : 2;
  const NSUInteger chunk_size = 64 + static_cast<NSUInteger>(ch) * static_cast<NSUInteger>(ch) * bytes;
  NSString* prev = @"x";
  for (int i = 0; i < depth; i++) {
    [m appendFormat:@"        tensor<%s, [%d, %d, 1, 1]> W%d = const()"
                    "[name = string(\"W%d\"), val = tensor<%s, [%d, %d, 1, 1]>"
                    "(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n",
                    dtype, ch, ch, i, i, dtype, ch, ch,
                    static_cast<unsigned long>(64 + static_cast<NSUInteger>(i) * chunk_size)];
    NSString* out = [NSString stringWithFormat:@"c%d", i];
    [m appendFormat:@"        tensor<%s, [1, %d, %d, %d]> %@ = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W%d, x = %@)[name = string(\"%@\")];\n",
                    dtype, ch, sp, sp, out, i, prev, out];
    prev = out;
  }
  [m appendFormat:@"    } -> (%@);\n}\n", prev];
  return m;
}

NSString* gen_mil_io_fp16_compute(const char* io_dtype, int ch, int sp, int depth) {
  NSMutableString* m = [NSMutableString stringWithString:mil_header()];
  [m appendFormat:@"    func main<ios18>(tensor<%s, [1, %d, %d, %d]> x) {\n", io_dtype, ch, sp, sp];
  append_conv_consts(m);
  [m appendString:@"        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"];
  [m appendFormat:@"        tensor<fp16, [1, %d, %d, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n",
                  ch, sp, sp];

  const NSUInteger chunk_size = 64 + static_cast<NSUInteger>(ch) * static_cast<NSUInteger>(ch) * 2u;
  NSString* prev = @"x16";
  for (int i = 0; i < depth; i++) {
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W%d = const()"
                    "[name = string(\"W%d\"), val = tensor<fp16, [%d, %d, 1, 1]>"
                    "(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n",
                    ch, ch, i, i, ch, ch,
                    static_cast<unsigned long>(64 + static_cast<NSUInteger>(i) * chunk_size)];
    NSString* out = [NSString stringWithFormat:@"c%d", i];
    [m appendFormat:@"        tensor<fp16, [1, %d, %d, %d]> %@ = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W%d, x = %@)[name = string(\"%@\")];\n",
                    ch, sp, sp, out, i, prev, out];
    prev = out;
  }
  [m appendFormat:@"        string to_io = const()[name = string(\"to_io\"), val = string(\"%s\")];\n",
                  io_dtype];
  [m appendFormat:@"        tensor<%s, [1, %d, %d, %d]> y = cast(dtype = to_io, x = %@)[name = string(\"cast_out\")];\n",
                  io_dtype, ch, sp, sp, prev];
  [m appendString:@"    } -> (y);\n}\n"];
  return m;
}

NSString* gen_mil_int8(int ch, int sp, int depth) {
  NSMutableString* m = [NSMutableString stringWithString:mil_header()];
  [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, %d, %d]> x) {\n", ch, sp, sp];
  append_conv_consts(m);
  [m appendString:@"        fp16 q_scale = const()[name = string(\"q_scale\"), val = fp16(0x1p-3)];\n"
                  @"        string q_dtype = const()[name = string(\"q_dtype\"), val = string(\"int8\")];\n"
                  @"        fp16 dq_scale = const()[name = string(\"dq_scale\"), val = fp16(0x1p-3)];\n"];

  const NSUInteger chunk_size = 64 + static_cast<NSUInteger>(ch) * static_cast<NSUInteger>(ch);
  NSString* prev = @"x";
  for (int i = 0; i < depth; i++) {
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W%d = constexpr_affine_dequantize()"
                    "[axis = int32(0), name = string(\"W%d\"), "
                    "quantized_data = tensor<int8, [%d, %d, 1, 1]>"
                    "(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu))), "
                    "scale = fp16(0x1p-3), zero_point = int8(0)];\n",
                    ch, ch, i, i, ch, ch,
                    static_cast<unsigned long>(64 + static_cast<NSUInteger>(i) * chunk_size)];
    NSString* conv_out = [NSString stringWithFormat:@"c%d", i];
    [m appendFormat:@"        tensor<fp16, [1, %d, %d, %d]> %@ = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W%d, x = %@)[name = string(\"%@\")];\n",
                    ch, sp, sp, conv_out, i, prev, conv_out];
    if (i < depth - 1) {
      NSString* q_out = [NSString stringWithFormat:@"q%d", i];
      [m appendFormat:@"        tensor<int8, [1, %d, %d, %d]> %@ = quantize(input = %@, output_dtype = q_dtype, scale = q_scale)[name = string(\"%@\")];\n",
                      ch, sp, sp, q_out, conv_out, q_out];
      NSString* dq_out = [NSString stringWithFormat:@"dq%d", i];
      [m appendFormat:@"        tensor<fp16, [1, %d, %d, %d]> %@ = dequantize(input = %@, scale = dq_scale)[name = string(\"%@\")];\n",
                      ch, sp, sp, dq_out, q_out, dq_out];
      prev = dq_out;
    } else {
      prev = conv_out;
    }
  }
  [m appendFormat:@"    } -> (%@);\n}\n", prev];
  return m;
}

NSString* gen_mil(NpuPrecision p, int ch, int sp, int depth) {
  if (p == NpuPrecision::INT8) return gen_mil_int8(ch, sp, depth);
  if (p == NpuPrecision::FP32) return gen_mil_io_fp16_compute("fp32", ch, sp, depth);
  if (p == NpuPrecision::BF16) return gen_mil_io_fp16_compute("bf16", ch, sp, depth);
  return gen_mil_typed(p, ch, sp, depth);
}

IOSurfaceRef create_surface(size_t bytes) {
  return IOSurfaceCreate((__bridge CFDictionaryRef)@{
      (id)kIOSurfaceWidth : @(bytes),
      (id)kIOSurfaceHeight : @1,
      (id)kIOSurfaceBytesPerElement : @1,
      (id)kIOSurfaceBytesPerRow : @(bytes),
      (id)kIOSurfaceAllocSize : @(bytes),
      (id)kIOSurfacePixelFormat : @0,
  });
}

bool init_private_ane(std::string& error) {
  void* h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
  if (!h) {
    error = "failed to dlopen AppleNeuralEngine.framework";
    return false;
  }
  if (!NSClassFromString(@"_ANEInMemoryModelDescriptor") ||
      !NSClassFromString(@"_ANEInMemoryModel") ||
      !NSClassFromString(@"_ANERequest") ||
      !NSClassFromString(@"_ANEIOSurfaceObject")) {
    error = "failed to resolve private ANE runtime classes";
    return false;
  }
  return true;
}

std::string ns_error(NSError* e) {
  if (!e) return "(nil)";
  return std::string([[e description] UTF8String]);
}

}  // namespace

bool run_npu_bench(const NpuBenchOptions& opt, NpuBenchResult& out, std::string& error) {
  @autoreleasepool {
    if (opt.channels <= 0 || opt.spatial <= 0 || opt.depth <= 0) {
      error = "channels/spatial/depth must be positive";
      return false;
    }
    if (opt.warmup < 0 || opt.repeats <= 0) {
      error = "warmup must be >= 0 and repeats must be > 0";
      return false;
    }
    if (!init_private_ane(error)) return false;

    const int ch = opt.channels;
    const int sp = opt.spatial;
    const int depth = opt.depth;
    NSData* mil_data = [gen_mil(opt.precision, ch, sp, depth) dataUsingEncoding:NSUTF8StringEncoding];
    const NpuPrecision weight_precision =
        (opt.precision == NpuPrecision::FP32 || opt.precision == NpuPrecision::BF16) ? NpuPrecision::FP16
                                                                                    : opt.precision;
    NSData* weights = build_weight_blob(weight_precision, ch, depth);
    if (!mil_data || !weights) {
      error = "failed to build MIL or weight blob";
      return false;
    }

    Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class Model = NSClassFromString(@"_ANEInMemoryModel");
    Class Request = NSClassFromString(@"_ANERequest");
    Class IO = NSClassFromString(@"_ANEIOSurfaceObject");

    NSError* e = nil;
    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        Desc, @selector(modelWithMILText:weights:optionsPlist:), mil_data,
        @{@"@model_path/weights/weight.bin" : @{@"offset" : @0, @"data" : weights}}, nil);
    if (!desc) {
      error = "modelWithMILText failed";
      return false;
    }

    id model = ((id(*)(Class, SEL, id))objc_msgSend)(Model, @selector(inMemoryModelWithDescriptor:), desc);
    if (!model) {
      error = "inMemoryModelWithDescriptor failed";
      return false;
    }

    id hx = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString* tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager* fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[tmp stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES
                         attributes:nil
                              error:nil];
    [mil_data writeToFile:[tmp stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [weights writeToFile:[tmp stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
      error = "ANE compile failed for " + std::string(precision_name(opt.precision)) +
              " (" + graph_kind_name(opt.precision) + "): " + ns_error(e);
      [fm removeItemAtPath:tmp error:nil];
      return false;
    }
    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
      error = "ANE load failed for " + std::string(precision_name(opt.precision)) + ": " + ns_error(e);
      [fm removeItemAtPath:tmp error:nil];
      return false;
    }

    const size_t io_bytes = static_cast<size_t>(ch) * static_cast<size_t>(sp) * static_cast<size_t>(sp) *
                            static_cast<size_t>(element_bytes(opt.precision));
    IOSurfaceRef input = create_surface(io_bytes);
    IOSurfaceRef output = create_surface(io_bytes);
    if (!input || !output) {
      error = "failed to allocate IOSurface";
      if (input) CFRelease(input);
      if (output) CFRelease(output);
      ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
      [fm removeItemAtPath:tmp error:nil];
      return false;
    }
    IOSurfaceLock(input, 0, nullptr);
    std::memset(IOSurfaceGetBaseAddress(input), 0, io_bytes);
    IOSurfaceUnlock(input, 0, nullptr);

    id wrapped_in = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(IO, @selector(objectWithIOSurface:), input);
    id wrapped_out = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(IO, @selector(objectWithIOSurface:), output);
    id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        Request, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[ wrapped_in ], @[ @0 ], @[ wrapped_out ], @[ @0 ], nil, nil, @0);
    if (!request) {
      error = "failed to create _ANERequest";
      CFRelease(input);
      CFRelease(output);
      ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
      [fm removeItemAtPath:tmp error:nil];
      return false;
    }

    for (int i = 0; i < opt.warmup; i++) {
      if (!((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(
              model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &e)) {
        error = "ANE warmup eval failed: " + ns_error(e);
        CFRelease(input);
        CFRelease(output);
        ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        [fm removeItemAtPath:tmp error:nil];
        return false;
      }
    }

    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < opt.repeats; i++) {
      const uint64_t t0 = mach_absolute_time();
      const BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(
          model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &e);
      const uint64_t t1 = mach_absolute_time();
      if (!ok) {
        error = "ANE eval failed: " + ns_error(e);
        CFRelease(input);
        CFRelease(output);
        ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        [fm removeItemAtPath:tmp error:nil];
        return false;
      }
      best = std::min(best, ticks_to_seconds(t1 - t0));
    }

    ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(input);
    CFRelease(output);
    [fm removeItemAtPath:tmp error:nil];

    const double ops = 2.0 * static_cast<double>(ch) * static_cast<double>(ch) *
                       static_cast<double>(sp) * static_cast<double>(sp) *
                       static_cast<double>(depth);
    out.best_seconds = best;
    out.operations = ops;
    out.score = ops / best / 1e12;
    out.npu_only = true;
    out.backend = "private-ane";
    out.note = "op=conv1x1-chain";
    if (opt.precision == NpuPrecision::INT8) {
      out.note += " | w8a8 quantize/dequantize";
    } else if (opt.precision == NpuPrecision::FP32) {
      out.note += " | fp32_io_fp16_compute";
    } else if (opt.precision == NpuPrecision::BF16) {
      out.note += " | bf16_io_fp16_compute";
    } else {
      out.note += " | dtype=" + std::string(precision_name(opt.precision));
    }
    out.note += " | direct _ANEInMemoryModel";
    {
      const double ref_peak = (opt.precision == NpuPrecision::INT8) ? 35.1 : 18.6;
      char util[64];
      std::snprintf(util, sizeof(util), " | util_ref_m4=%.1f%%", out.score / ref_peak * 100.0);
      out.note += util;
    }
    return true;
  }
}
