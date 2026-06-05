#import "MTFLOPSAppDelegate.h"

#import "MTFLOPSViewController.h"

@implementation MTFLOPSAppDelegate

- (BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)launchOptions {
  (void)application;
  (void)launchOptions;
  self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
  self.window.rootViewController = [[MTFLOPSViewController alloc] init];
  [self.window makeKeyAndVisible];
  return YES;
}

@end
