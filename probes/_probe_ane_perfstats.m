#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dlfcn.h>

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

static NSData* weight_blob(int ch) {
  const NSUInteger wsize = (NSUInteger)ch * (NSUInteger)ch * 2u;
  const NSUInteger total = 64u + 64u + wsize;
  uint8_t* buf = (uint8_t*)calloc(total, 1);
  buf[0] = 1;
  buf[4] = 2;
  uint8_t* chunk = buf + 64;
  chunk[0] = 0xEF;
  chunk[1] = 0xBE;
  chunk[2] = 0xAD;
  chunk[3] = 0xDE;
  chunk[4] = 1;
  chunk[10] = 0x10;
  _Float16* w = (_Float16*)(chunk + 64);
  for (int i = 0; i < ch; i++) w[i * ch + i] = (_Float16)1.0f;
  return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSString* mil_text(int ch, int sp) {
  return [NSString stringWithFormat:
      @"program(1.3)\n"
       "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
       "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
       "{\"coremltools-version\", \"9.0\"}})]\n"
       "{\n"
       "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
       "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
       "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
       "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
       "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
       "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
       "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
       "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
       "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
       "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
       "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)[name=string(\"conv\")];\n"
       "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
       "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
       "    } -> (y);\n"
       "}\n",
      ch, sp, ch, sp, ch, ch, ch, ch, ch, sp, ch, sp];
}

static void print_stats(id obj, const char* label) {
  printf("%s: %s\n", label, obj ? [[obj description] UTF8String] : "nil");
  if (!obj) return;
  @try {
    unsigned long long hw = ((unsigned long long(*)(id, SEL))objc_msgSend)(obj, @selector(hwExecutionTime));
    id pc = ((id(*)(id, SEL))objc_msgSend)(obj, @selector(perfCounterData));
    id raw = ((id(*)(id, SEL))objc_msgSend)(obj, @selector(pStatsRawData));
    printf("  hwExecutionTime=%llu perfCounterData=%s pStatsRawData=%s\n", hw,
           pc ? [[pc description] UTF8String] : "nil",
           raw ? [[raw description] UTF8String] : "nil");
  } @catch (NSException* e) {
    printf("  stats read exception: %s\n", [[e reason] UTF8String]);
  }
}

int main() {
  @autoreleasepool {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class Model = NSClassFromString(@"_ANEInMemoryModel");
    Class Request = NSClassFromString(@"_ANERequest");
    Class IO = NSClassFromString(@"_ANEIOSurfaceObject");
    Class Perf = NSClassFromString(@"_ANEPerformanceStats");
    Class PerfSurf = NSClassFromString(@"_ANEPerformanceStatsIOSurface");
    if (!Desc || !Model || !Request || !IO || !Perf || !PerfSurf) {
      printf("missing classes\n");
      return 2;
    }

    const int ch = 64, sp = 32;
    NSData* md = [mil_text(ch, sp) dataUsingEncoding:NSUTF8StringEncoding];
    NSData* wb = weight_blob(ch);
    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        Desc, @selector(modelWithMILText:weights:optionsPlist:), md,
        @{@"@model_path/weights/weight.bin" : @{@"offset" : @0, @"data" : wb}}, nil);
    id model = ((id(*)(Class, SEL, id))objc_msgSend)(Model, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString* tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[tmp stringByAppendingPathComponent:@"weights"]
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];
    [md writeToFile:[tmp stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wb writeToFile:[tmp stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError* e = nil;
    ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &e);

    const size_t bytes = (size_t)ch * (size_t)sp * 4u;
    IOSurfaceRef in = make_surface(bytes);
    IOSurfaceRef out = make_surface(bytes);
    id wi = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(IO, @selector(objectWithIOSurface:), in);
    id wo = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(IO, @selector(objectWithIOSurface:), out);

    const unsigned masks[] = {0, 1, 3, 0x7, 0xff, 0xffffffffu};
    for (unsigned mi = 0; mi < sizeof(masks) / sizeof(masks[0]); mi++) {
      unsigned mask = masks[mi];
      id perf = ((id(*)(Class, SEL, unsigned long long))objc_msgSend)(
          Perf, @selector(statsWithHardwareExecutionNS:), 0ULL);
      ((void(*)(id, SEL, unsigned int))objc_msgSend)(model, @selector(setPerfStatsMask:), mask);
      id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
          Request, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
          @[ wi ], @[ @0 ], @[ wo ], @[ @0 ], nil, @[ perf ], @0);
      BOOL ok = NO;
      @try {
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        printf("\nmask=0x%x eval=%s\n", mask, ok ? "OK" : [[e description] UTF8String]);
      } @catch (NSException* ex) {
        printf("\nmask=0x%x eval exception=%s\n", mask, [[ex reason] UTF8String]);
      }
      print_stats(perf, "perf object");
      print_stats(((id(*)(id, SEL))objc_msgSend)(req, @selector(perfStats)), "request.perfStats");
      id arr = ((id(*)(id, SEL))objc_msgSend)(req, @selector(perfStatsArray));
      printf("request.perfStatsArray: %s\n", arr ? [[arr description] UTF8String] : "nil");
    }

    printf("\n=== _ANEPerformanceStatsIOSurface attempts ===\n");
    const long stat_types[] = {0, 1, 2, 3, 4, 5};
    for (unsigned si = 0; si < sizeof(stat_types) / sizeof(stat_types[0]); si++) {
      long stat_type = stat_types[si];
      IOSurfaceRef stats_surface = make_surface(65536);
      id stats_io = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(IO, @selector(objectWithIOSurface:), stats_surface);
      id perf_item = ((id(*)(Class, SEL, id, long))objc_msgSend)(
          PerfSurf, @selector(objectWithIOSurface:statType:), stats_io, stat_type);
      ((void(*)(id, SEL, unsigned int))objc_msgSend)(model, @selector(setPerfStatsMask:), 0xffffffffu);
      id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
          Request, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
          @[ wi ], @[ @0 ], @[ wo ], @[ @0 ], nil, @[ perf_item ], @0);
      BOOL ok = NO;
      @try {
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        printf("\nstatType=%ld eval=%s\n", stat_type, ok ? "OK" : [[e description] UTF8String]);
      } @catch (NSException* ex) {
        printf("\nstatType=%ld eval exception=%s\n", stat_type, [[ex reason] UTF8String]);
      }
      print_stats(((id(*)(id, SEL))objc_msgSend)(req, @selector(perfStats)), "request.perfStats");
      id arr = ((id(*)(id, SEL))objc_msgSend)(req, @selector(perfStatsArray));
      printf("request.perfStatsArray: %s\n", arr ? [[arr description] UTF8String] : "nil");
      IOSurfaceLock(stats_surface, 0, NULL);
      const uint8_t* bytes_ptr = (const uint8_t*)IOSurfaceGetBaseAddress(stats_surface);
      size_t nonzero = 0;
      size_t first_nonzero = (size_t)-1;
      for (size_t bi = 0; bi < 65536; bi++) {
        if (bytes_ptr[bi] != 0) {
          nonzero++;
          if (first_nonzero == (size_t)-1) first_nonzero = bi;
        }
      }
      printf("stats surface nonzero=%zu first_nonzero=%zd\n", nonzero,
             first_nonzero == (size_t)-1 ? -1 : (ssize_t)first_nonzero);
      printf("stats surface first 32 bytes:");
      for (int bi = 0; bi < 32; bi++) printf(" %02x", bytes_ptr[bi]);
      printf("\n");
      IOSurfaceUnlock(stats_surface, 0, NULL);
      CFRelease(stats_surface);
    }

    ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
    [[NSFileManager defaultManager] removeItemAtPath:tmp error:nil];
    CFRelease(in);
    CFRelease(out);
  }
  return 0;
}
