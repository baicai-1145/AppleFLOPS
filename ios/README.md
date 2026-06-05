# MTFLOPS iOS Runner

This directory contains a minimal iPhone runner for measuring local CPU, GPU, and ANE/NPU compute from the same benchmark codebase.

## Build

Compile-only check, no signing or install:

```sh
xcodebuild -project ios/MTFLOPSRunner.xcodeproj \
  -scheme MTFLOPSRunner \
  -configuration Release \
  -sdk iphoneos \
  CODE_SIGNING_ALLOWED=NO \
  build
```

Build for the connected iPhone:

```sh
xcodebuild -project ios/MTFLOPSRunner.xcodeproj \
  -scheme MTFLOPSRunner \
  -configuration Release \
  -destination 'id=<iphone-udid>' \
  -allowProvisioningUpdates \
  build
```

The default bundle id is `com.example.mtflops.runner`. For a real device build, set your own bundle identifier and development team in Xcode, or pass signing settings on the `xcodebuild` command line.

## Run

Open `ios/MTFLOPSRunner.xcodeproj` in Xcode, select the connected iPhone, then run the `MTFLOPSRunner` scheme. The app starts benchmarks automatically and shows a text report.

If iOS refuses to launch a freshly installed development build, trust the developer profile on the phone first:

```text
Settings -> General -> VPN & Device Management -> Apple Development -> Trust
```

The app also writes:

```text
Documents/mtflops-ios-results.json
```

## Coverage

- CPU: SME MOPA peak probes for FP32, FP16, BF16, and INT8 when SME is available, with NEON/BF16/I8MM fallback probes for older devices.
- GPU: Metal simdgroup peak workload for FP32, FP16, and BF16 using `shaders/gemm.metal`.
- NPU/ANE: private `_ANEInMemoryModel` conv1x1-chain workload for FP16 and INT8.

## Limitations

- CPU SME availability depends on the iPhone CPU generation and iOS toolchain/runtime support.
- GPU INT8 is deliberately not recorded as trusted matrix-unit peak.
- NPU FP32 and BF16 are deliberately not recorded as native scores.
- iPhone power is not measured here; macOS `powermetrics` cannot report iPhone package power.
- The ANE path uses private AppleNeuralEngine APIs and is intended only for local research/dev-signed runs.
