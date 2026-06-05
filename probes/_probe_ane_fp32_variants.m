#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static NSString* header(void) {
  return @"program(1.3)\n"
          "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
          "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
          "{\"coremltools-version\", \"9.0\"}})]\n"
          "{\n";
}

static NSString* common_consts(void) {
  return @"        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
          @"        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
          @"        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
          @"        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
          @"        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n";
}

static NSData* weight_blob(NSString* dtype, int ch) {
  const BOOL is_fp32 = [dtype isEqualToString:@"fp32"];
  const int bytes = is_fp32 ? 4 : 2;
  const NSUInteger wsize = (NSUInteger)ch * (NSUInteger)ch * bytes;
  const NSUInteger total = 64u + 64u + wsize;
  uint8_t* buf = (uint8_t*)calloc(total, 1);
  if (!buf) return nil;
  buf[0] = 1;
  buf[4] = 2;
  uint8_t* chunk = buf + 64;
  chunk[0] = 0xEF;
  chunk[1] = 0xBE;
  chunk[2] = 0xAD;
  chunk[3] = 0xDE;
  chunk[4] = 1;
  chunk[10] = is_fp32 ? 0x20 : 0x10;

  if (is_fp32) {
    float* w = (float*)(chunk + 64);
    for (int i = 0; i < ch * ch; ++i) w[i] = 0.0f;
    for (int i = 0; i < ch; ++i) w[i * ch + i] = 1.0003f;
  } else {
    _Float16* w = (_Float16*)(chunk + 64);
    for (int i = 0; i < ch * ch; ++i) w[i] = (_Float16)0.0f;
    for (int i = 0; i < ch; ++i) w[i * ch + i] = (_Float16)1.0003f;
  }
  return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSString* mil_true_typed(NSString* dtype, NSString* target, int ch, int sp) {
  NSMutableString* m = [NSMutableString stringWithString:header()];
  [m appendFormat:@"    func main<%@>(tensor<%@, [1,%d,%d,%d]> x) {\n", target, dtype, ch, sp, sp];
  [m appendString:common_consts()];
  [m appendFormat:@"        tensor<%@, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<%@, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n",
                  dtype, ch, ch, dtype, ch, ch];
  [m appendFormat:@"        tensor<%@, [1,%d,%d,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"conv\")];\n",
                  dtype, ch, sp, sp];
  [m appendString:@"    } -> (y);\n}\n"];
  return m;
}

static NSString* mil_fp32_io_fp16_compute(NSString* target, int ch, int sp) {
  NSMutableString* m = [NSMutableString stringWithString:header()];
  [m appendFormat:@"    func main<%@>(tensor<fp32, [1,%d,%d,%d]> x) {\n", target, ch, sp, sp];
  [m appendString:common_consts()];
  [m appendString:@"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"];
  [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cast_in\")];\n", ch, sp, sp];
  [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n",
                  ch, ch, ch, ch];
  [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)[name=string(\"conv\")];\n",
                  ch, sp, sp];
  [m appendString:@"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"];
  [m appendFormat:@"        tensor<fp32, [1,%d,%d,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cast_out\")];\n", ch, sp, sp];
  [m appendString:@"    } -> (y);\n}\n"];
  return m;
}

static NSString* mil_mismatch(NSString* target, int ch, int sp) {
  NSMutableString* m = [NSMutableString stringWithString:header()];
  [m appendFormat:@"    func main<%@>(tensor<fp32, [1,%d,%d,%d]> x) {\n", target, ch, sp, sp];
  [m appendString:common_consts()];
  [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n",
                  ch, ch, ch, ch];
  [m appendFormat:@"        tensor<fp32, [1,%d,%d,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"conv\")];\n",
                  ch, sp, sp];
  [m appendString:@"    } -> (y);\n}\n"];
  return m;
}

static IOSurfaceRef make_surface(size_t bytes) {
  return IOSurfaceCreate((__bridge CFDictionaryRef)@{
      (id)kIOSurfaceWidth : @(bytes),
      (id)kIOSurfaceHeight : @1,
      (id)kIOSurfaceBytesPerElement : @1,
      (id)kIOSurfaceBytesPerRow : @(bytes),
      (id)kIOSurfaceAllocSize : @(bytes),
      (id)kIOSurfacePixelFormat : @0,
  });
}

static NSString* one_line(NSError* e) {
  if (!e) return @"(nil)";
  return [[e description] stringByReplacingOccurrencesOfString:@"\n" withString:@" "];
}

static void list_tmp(NSString* tmp) {
  NSDirectoryEnumerator* en = [[NSFileManager defaultManager] enumeratorAtPath:tmp];
  NSString* p;
  while ((p = [en nextObject])) {
    NSString* full = [tmp stringByAppendingPathComponent:p];
    NSDictionary* attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:full error:nil];
    if ([[attrs fileType] isEqualToString:NSFileTypeRegular]) {
      printf("    file %s size=%llu\n", [p UTF8String], (unsigned long long)[attrs fileSize]);
    } else {
      printf("    dir  %s\n", [p UTF8String]);
    }
  }
}

static void copy_cached_model(NSString* hx, NSString* variant, NSString* target) {
  NSString* root = @"/Library/Caches/com.apple.aned";
  NSString* out_dir = @"/Volumes/2T/_ane_reverse_tmp/ane_model_cache_copies";
  [[NSFileManager defaultManager] createDirectoryAtPath:out_dir
                            withIntermediateDirectories:YES
                                             attributes:nil
                                                  error:nil];
  NSDirectoryEnumerator* en = [[NSFileManager defaultManager] enumeratorAtPath:root];
  NSString* rel;
  BOOL found = NO;
  while ((rel = [en nextObject])) {
    if (![rel containsString:hx] || ![[rel lastPathComponent] isEqualToString:@"model.hwx"]) continue;
    found = YES;
    NSString* src = [root stringByAppendingPathComponent:rel];
    NSString* dst_name = [NSString stringWithFormat:@"%@-%@-%@.hwx", variant, target, hx];
    NSString* dst = [out_dir stringByAppendingPathComponent:dst_name];
    [[NSFileManager defaultManager] removeItemAtPath:dst error:nil];
    NSError* copy_err = nil;
    BOOL copied = [[NSFileManager defaultManager] copyItemAtPath:src toPath:dst error:&copy_err];
    printf("    cache_copy=%s src=%s dst=%s%s\n", copied ? "OK" : "FAIL",
           [src UTF8String], [dst UTF8String],
           copied ? "" : [[one_line(copy_err) stringByAppendingString:@""] UTF8String]);
  }
  if (!found) {
    printf("    cache_copy=NOT_FOUND_OR_NO_PERMISSION root=%s\n", [root UTF8String]);
  }
}

static BOOL try_variant(NSString* variant, NSString* target, BOOL eval_if_ok) {
  const int ch = 16, sp = 16;
  NSString* weight_dtype = @"fp16";
  NSString* mil = nil;
  if ([variant isEqualToString:@"true_fp32"]) {
    weight_dtype = @"fp32";
    mil = mil_true_typed(@"fp32", target, ch, sp);
  } else if ([variant isEqualToString:@"true_fp16"]) {
    weight_dtype = @"fp16";
    mil = mil_true_typed(@"fp16", target, ch, sp);
  } else if ([variant isEqualToString:@"fp32_io_fp16_compute"]) {
    weight_dtype = @"fp16";
    mil = mil_fp32_io_fp16_compute(target, ch, sp);
  } else if ([variant isEqualToString:@"fp32_input_fp16_weight_no_cast"]) {
    weight_dtype = @"fp16";
    mil = mil_mismatch(target, ch, sp);
  } else {
    printf("unknown variant %s\n", [variant UTF8String]);
    return NO;
  }

  NSData* md = [mil dataUsingEncoding:NSUTF8StringEncoding];
  NSData* wb = weight_blob(weight_dtype, ch);
  Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
  Class Model = NSClassFromString(@"_ANEInMemoryModel");
  Class Request = NSClassFromString(@"_ANERequest");
  Class IO = NSClassFromString(@"_ANEIOSurfaceObject");
  id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
      Desc, @selector(modelWithMILText:weights:optionsPlist:), md,
      @{@"@model_path/weights/weight.bin" : @{@"offset" : @0, @"data" : wb}}, nil);
  id model = desc ? ((id(*)(Class, SEL, id))objc_msgSend)(Model, @selector(inMemoryModelWithDescriptor:), desc) : nil;
  id hx = model ? ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier)) : nil;
  NSString* tmp = hx ? [NSTemporaryDirectory() stringByAppendingPathComponent:hx] : nil;
  if (!desc || !model || !tmp) {
    printf("%-30s target=%-5s setup=FAIL desc/model/tmp missing\n", [variant UTF8String], [target UTF8String]);
    return NO;
  }
  [[NSFileManager defaultManager] createDirectoryAtPath:[tmp stringByAppendingPathComponent:@"weights"]
                            withIntermediateDirectories:YES
                                             attributes:nil
                                                  error:nil];
  [md writeToFile:[tmp stringByAppendingPathComponent:@"model.mil"] atomically:YES];
  [wb writeToFile:[tmp stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

  NSError* e = nil;
  BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
      model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
  printf("%-30s target=%-5s compile=%s tmp=%s\n", [variant UTF8String], [target UTF8String],
         ok ? "OK" : "FAIL", [tmp UTF8String]);
  if (!ok) {
    printf("    error: %s\n", [one_line(e) UTF8String]);
    [[NSFileManager defaultManager] removeItemAtPath:tmp error:nil];
    return NO;
  }
  list_tmp(tmp);

  if (eval_if_ok && ([variant isEqualToString:@"true_fp32"] || [variant isEqualToString:@"fp32_io_fp16_compute"])) {
    e = nil;
    BOOL loaded = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    printf("    load=%s%s\n", loaded ? "OK" : "FAIL ", loaded ? "" : [one_line(e) UTF8String]);
    if (loaded) {
      copy_cached_model(hx, variant, target);
      const size_t bytes = (size_t)ch * (size_t)sp * (size_t)sp * sizeof(float);
      IOSurfaceRef in = make_surface(bytes);
      IOSurfaceRef out = make_surface(bytes);
      IOSurfaceLock(in, 0, NULL);
      float* x = (float*)IOSurfaceGetBaseAddress(in);
      for (int i = 0; i < ch * sp * sp; ++i) x[i] = 1.0003f + (float)i * 0.00001f;
      IOSurfaceUnlock(in, 0, NULL);
      id wi = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(IO, @selector(objectWithIOSurface:), in);
      id wo = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(IO, @selector(objectWithIOSurface:), out);
      id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
          Request, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
          @[ wi ], @[ @0 ], @[ wo ], @[ @0 ], nil, nil, @0);
      e = nil;
      BOOL ev = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(
          model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
      printf("    eval=%s%s\n", ev ? "OK" : "FAIL ", ev ? "" : [one_line(e) UTF8String]);
      CFRelease(in);
      CFRelease(out);
      ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
    }
  }

  [[NSFileManager defaultManager] removeItemAtPath:tmp error:nil];
  return YES;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    NSArray* variants = @[
      @"true_fp16",
      @"true_fp32",
      @"fp32_io_fp16_compute",
      @"fp32_input_fp16_weight_no_cast",
    ];
    NSArray* targets = @[ @"ios18", @"ios19" ];
    const BOOL eval_if_ok = argc > 1 && strcmp(argv[1], "--eval") == 0;
    for (NSString* target in targets) {
      for (NSString* variant in variants) {
        try_variant(variant, target, eval_if_ok);
      }
    }
  }
  return 0;
}
