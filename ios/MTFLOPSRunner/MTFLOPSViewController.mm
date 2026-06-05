#import "MTFLOPSViewController.h"

#import "IOSBenchRunner.h"

@interface MTFLOPSViewController ()
@property(nonatomic, strong) UITextView* textView;
@property(nonatomic, strong) UIButton* runButton;
@property(nonatomic, assign) BOOL running;
@end

@implementation MTFLOPSViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  self.view.backgroundColor = UIColor.systemBackgroundColor;

  self.textView = [[UITextView alloc] initWithFrame:CGRectZero];
  self.textView.translatesAutoresizingMaskIntoConstraints = NO;
  self.textView.editable = NO;
  self.textView.font = [UIFont monospacedSystemFontOfSize:12.0 weight:UIFontWeightRegular];
  self.textView.text = @"Ready\n";
  [self.view addSubview:self.textView];

  self.runButton = [UIButton buttonWithType:UIButtonTypeSystem];
  self.runButton.translatesAutoresizingMaskIntoConstraints = NO;
  [self.runButton setTitle:@"Run Benchmark" forState:UIControlStateNormal];
  [self.runButton addTarget:self action:@selector(runBenchmark) forControlEvents:UIControlEventTouchUpInside];
  [self.view addSubview:self.runButton];

  UILayoutGuide* guide = self.view.safeAreaLayoutGuide;
  [NSLayoutConstraint activateConstraints:@[
    [self.runButton.topAnchor constraintEqualToAnchor:guide.topAnchor constant:12.0],
    [self.runButton.leadingAnchor constraintEqualToAnchor:guide.leadingAnchor constant:16.0],
    [self.runButton.trailingAnchor constraintEqualToAnchor:guide.trailingAnchor constant:-16.0],
    [self.runButton.heightAnchor constraintEqualToConstant:44.0],
    [self.textView.topAnchor constraintEqualToAnchor:self.runButton.bottomAnchor constant:12.0],
    [self.textView.leadingAnchor constraintEqualToAnchor:guide.leadingAnchor constant:12.0],
    [self.textView.trailingAnchor constraintEqualToAnchor:guide.trailingAnchor constant:-12.0],
    [self.textView.bottomAnchor constraintEqualToAnchor:guide.bottomAnchor constant:-12.0],
  ]];

  dispatch_async(dispatch_get_main_queue(), ^{
    [self runBenchmark];
  });
}

- (void)appendLine:(NSString*)line {
  NSString* current = self.textView.text ?: @"";
  self.textView.text = [current stringByAppendingFormat:@"%@\n", line];
  NSRange bottom = NSMakeRange(self.textView.text.length, 0);
  [self.textView scrollRangeToVisible:bottom];
}

- (void)runBenchmark {
  if (self.running) return;
  self.running = YES;
  self.runButton.enabled = NO;
  self.textView.text = @"Running...\n";

  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    NSDictionary* result = RunIOSBenchmarks(^(NSString* text) {
      dispatch_async(dispatch_get_main_queue(), ^{
        [self appendLine:text];
      });
    });

    NSError* error = nil;
    NSURL* url = WriteBenchmarkResult(result, &error);
    NSString* report = FormatBenchmarkReport(result);
    NSLog(@"\n%@", report);
    if (url) NSLog(@"MTFLOPS result JSON: %@", url.path);
    else NSLog(@"Failed to write MTFLOPS result JSON: %@", error);

    dispatch_async(dispatch_get_main_queue(), ^{
      self.textView.text = report;
      if (url) [self appendLine:[NSString stringWithFormat:@"\nJSON: %@", url.path]];
      else [self appendLine:[NSString stringWithFormat:@"\nJSON write failed: %@", error]];
      self.running = NO;
      self.runButton.enabled = YES;
      [self.runButton setTitle:@"Run Again" forState:UIControlStateNormal];
    });
  });
}

@end
