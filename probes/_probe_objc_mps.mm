#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <objc/runtime.h>
#import <dlfcn.h>

static void dumpClassMethods(const char *className) {
    Class cls = objc_getClass(className);
    if (!cls) {
        printf("class-not-found %s\n", className);
        return;
    }
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    printf("class %s methods %u\n", className, count);
    for (unsigned int i = 0; i < count; ++i) {
        SEL sel = method_getName(methods[i]);
        IMP imp = method_getImplementation(methods[i]);
        Dl_info info = {};
        dladdr((const void *)imp, &info);
        printf("- %s\t%p\t%s\t%s\n",
               sel_getName(sel),
               imp,
               info.dli_fname ? info.dli_fname : "?",
               info.dli_sname ? info.dli_sname : "?");
    }
    free(methods);
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("device=%s\n", device.name.UTF8String);
        dumpClassMethods("MPSNDArrayQuantizedMatrixMultiplication");
        dumpClassMethods("MPSNDArrayMatrixMultiplication");
        dumpClassMethods("MPSNDArrayAffineQuantizationDescriptor");
    }
    return 0;
}
