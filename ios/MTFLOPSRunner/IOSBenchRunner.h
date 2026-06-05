#import <Foundation/Foundation.h>

typedef void (^MTFLOPSProgressBlock)(NSString* text);

NSDictionary* RunIOSBenchmarks(MTFLOPSProgressBlock progress);
NSString* FormatBenchmarkReport(NSDictionary* result);
NSURL* WriteBenchmarkResult(NSDictionary* result, NSError** error);
