#import <Foundation/Foundation.h>

typedef void (^AppleFLOPSProgressBlock)(NSString* text);

NSDictionary* RunIOSBenchmarks(AppleFLOPSProgressBlock progress);
NSString* FormatBenchmarkReport(NSDictionary* result);
NSURL* WriteBenchmarkResult(NSDictionary* result, NSError** error);
