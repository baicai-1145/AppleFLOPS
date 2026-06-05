#import "AppleFLOPSAppDelegate.h"

#import "AppleFLOPSViewController.h"

@implementation AppleFLOPSAppDelegate

- (BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)launchOptions {
  (void)application;
  (void)launchOptions;
  self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
  self.window.rootViewController = [[AppleFLOPSViewController alloc] init];
  [self.window makeKeyAndVisible];
  return YES;
}

@end
