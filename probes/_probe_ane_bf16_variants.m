#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>

static NSData* blob(int ch, BOOL bf16) {
  const int bytes = bf16 ? 2 : 2;
  const NSUInteger wsize = (NSUInteger)ch * (NSUInteger)ch * bytes;
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
  uint16_t* w = (uint16_t*)(chunk + 64);
  for (int i = 0; i < ch; i++) {
    if (bf16) {
      w[i * ch + i] = 0x3f80;  // bf16(1.0)
    } else {
      ((_Float16*)w)[i * ch + i] = (_Float16)1.0f;
    }
  }
  return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

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

static NSString* mil(NSString* variant, NSString* target, int ch, int sp) {
  NSMutableString* m = [NSMutableString stringWithString:header()];
  if ([variant isEqualToString:@"true_bf16"]) {
    [m appendFormat:@"    func main<%@>(tensor<bf16, [1,%d,1,%d]> x) {\n", target, ch, sp];
    [m appendString:common_consts()];
    [m appendFormat:@"        tensor<bf16, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<bf16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n", ch, ch, ch, ch];
    [m appendFormat:@"        tensor<bf16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"conv\")];\n", ch, sp];
    [m appendString:@"    } -> (y);\n}\n"];
  } else if ([variant isEqualToString:@"bf16_io_fp16"]) {
    [m appendFormat:@"    func main<%@>(tensor<bf16, [1,%d,1,%d]> x) {\n", target, ch, sp];
    [m appendString:common_consts()];
    [m appendString:@"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n", ch, sp];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n", ch, ch, ch, ch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)[name=string(\"conv\")];\n", ch, sp];
    [m appendString:@"        string tobf = const()[name=string(\"tobf\"), val=string(\"bf16\")];\n"];
    [m appendFormat:@"        tensor<bf16, [1,%d,1,%d]> y = cast(dtype=tobf,x=y16)[name=string(\"cout\")];\n", ch, sp];
    [m appendString:@"    } -> (y);\n}\n"];
  } else {
    [m appendFormat:@"    func main<%@>(tensor<fp32, [1,%d,1,%d]> x) {\n", target, ch, sp];
    [m appendString:common_consts()];
    [m appendString:@"        string tobf = const()[name=string(\"tobf\"), val=string(\"bf16\")];\n"];
    [m appendFormat:@"        tensor<bf16, [1,%d,1,%d]> xb = cast(dtype=tobf,x=x)[name=string(\"cbf\")];\n", ch, sp];
    [m appendString:@"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=xb)[name=string(\"cin\")];\n", ch, sp];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n", ch, ch, ch, ch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)[name=string(\"conv\")];\n", ch, sp];
    [m appendString:@"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n", ch, sp];
    [m appendString:@"    } -> (y);\n}\n"];
  }
  return m;
}

static BOOL try_compile(NSString* variant, NSString* target) {
  const int ch = 16, sp = 16;
  NSData* md = [mil(variant, target, ch, sp) dataUsingEncoding:NSUTF8StringEncoding];
  NSData* wb = blob(ch, [variant isEqualToString:@"true_bf16"]);
  Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
  Class Model = NSClassFromString(@"_ANEInMemoryModel");
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
  BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
      model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
  printf("%-16s target=%-5s compile=%s%s\n", [variant UTF8String], [target UTF8String],
         ok ? "OK" : "FAIL ", ok ? "" : [[[e description] stringByReplacingOccurrencesOfString:@"\n" withString:@" "] UTF8String]);
  [[NSFileManager defaultManager] removeItemAtPath:tmp error:nil];
  return ok;
}

int main() {
  @autoreleasepool {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    NSArray* variants = @[ @"true_bf16", @"bf16_io_fp16", @"fp32_bf16_fp16" ];
    NSArray* targets = @[ @"ios18", @"ios19" ];
    for (NSString* v in variants) {
      for (NSString* t in targets) {
        try_compile(v, t);
      }
    }
  }
  return 0;
}
