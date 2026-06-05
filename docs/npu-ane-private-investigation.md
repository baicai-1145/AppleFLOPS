# NPU / ANE Private API Investigation

This project uses the private Apple Neural Engine runtime path inspired by
`maderix/ANE`:

- `AppleNeuralEngine.framework`
- `_ANEInMemoryModelDescriptor`
- `_ANEInMemoryModel`
- `_ANERequest`
- `_ANEIOSurfaceObject`

The benchmark workload is a 1x1 conv chain. For `N=512`, `spatial=64`, and
`depth=128`, FP16 and INT8 reach M4 reference peak throughput.

## Runtime / Compiler / TD Path

Static disassembly on this MacBook Air M4 shows the private ANE path at these
layers:

```text
MIL text / _ANEInMemoryModelDescriptor
  -> _ANEInMemoryModel compileWithQoS
  -> _ANEClient / _ANEDaemonConnection compileModel
  -> ANECompilerService.xpc
  -> ANECompiler ANECCompile / ANECCompileOnline
  -> MILOpConverter / ne_conv / ZinNEConvLayer / ZinAneTd<20u>
  -> .hwx / model.hwx
  -> _ANEInMemoryModel loadWithQoS
  -> programHandle / intermediateBufferHandle / queueDepth / program
  -> _ANEInMemoryModel evaluateWithQoS
  -> _ANEProgramForEvaluation processRequest
  -> ANEProgramProcessRequestDirect / ANEDeviceOpen
  -> H1xANELoadBalancer / H11ANEIn driver path
```

Runtime evidence from `AppleNeuralEngine.framework`:

- `-[_ANEInMemoryModel compileWithQoS:options:error:]` saves model files,
  creates an ANE model object, builds compiler options, then calls
  `_objc_msgSend$compileModel:options:qos:error:`.
- `-[_ANEInMemoryModel loadWithQoS:options:error:]` calls
  `_objc_msgSend$loadModel:options:qos:error:`. On success it copies
  `programHandle`, `intermediateBufferHandle`, `queueDepth`, `modelAttributes`,
  `perfStatsMask`, and `program` from the loaded model and sets model state to
  loaded.
- `-[_ANEInMemoryModel evaluateWithQoS:options:request:error:]` either calls
  `_ANEClient evaluateWithModel:options:request:qos:error:` or, on the local
  direct path, gets the loaded `program` and calls
  `_objc_msgSend$processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:`.
- `-[_ANEProgramForEvaluation processRequest:...]` fills request structures
  with `programHandle`. Its strings reference `ANEProgramProcessRequestDirect()`
  and the same framework contains `ANEProgramMemoryMapRequest()` and
  `ANEProgramMemoryUnMapRequest()`.
- The framework strings include `aned`, `com.apple.aneuserd`,
  `/Library/Caches/com.apple.aned`, `/Library/Caches/com.apple.aneuserd`,
  `model.hwx`, `.hwx`, `programHandle`, `intermediateBufferHandle`, and
  `queueDepth`.

Compiler service evidence:

- `ANECompilerService.xpc` links against `AppleNeuralEngine.framework`,
  `ANECompiler.framework`, `MIL.framework`, and `Espresso.framework`.
- Its strings include `_ANEMILCompiler`, `_ANEMLIRCompiler`,
  `_ANECompiler : ANECCompile() FAILED`, `Calling ANECCompileOnline...`,
  `MODELCACHEDIR`, `modelBinaryName`, `cachedModelPathFor:csIdentity:`,
  `compileModelAt:... outputURL ... aotModelBinaryPath ...`, and
  `ANECCompile(%@) FAILED: err=%@`.

ANECompiler lowering/codegen evidence:

- Symbols include `MILOpConverter::NEConv(...)`,
  `ZinMirNEConvUnit::CreateLayer(...)`, `ZinIrNEConvUnit::CreateLayer(...)`,
  `ZinPatternUtils::ToNEConv(...)`, `ZinMirMatMul::LowerNEMatMulToNEConv(...)`,
  and `ZinMirMatMul::ConvertMatMulToNEConv(...)`.
- Strings include `ne_conv` and `ane.ne_conv`.
- `ZinAneTd<20u>::HandleANELayer(...)` checks layer kind and dispatches NE
  layers through `HandleNELayer<20u>(...)`.
- `ZinAneTd<20u>::HandleNEConfig(...)` calls:
  `GetMacCfgOpMode -> SetOpMode`,
  `GetMacCfgKernelMode -> SetKernelMode`,
  `GetKernelCfgKernelFmt -> SetKernelFmt`,
  then programs binary point, postscale, bias, matrix-vector bias, asymmetric
  quantization, zero-detection, double-INT8 enable, sparse, and palette fields.
- `ZinAneTd<20u>::HandleCommonConvOpcode(...)` programs conv geometry into the
  TD: output width/height, kernel width/height, padding, stride, dilation/group
  related fields, and unicast fields.

The v20 TD setters directly write hardware task descriptor fields:

- `SetCommonInFmt(ZinTensorFormat)` accepts enum values `1`, `2`, `3`, and
  `0xc`, then writes bits `0..2` at `[x0,#0x228]`.
- `SetCommonSrc2InFmt(ZinTensorFormat)` accepts the same enum values and writes
  bits `3..5` at `[x0,#0x228]`.
- `SetCommonOutFmt(ZinTensorFormat)` writes output format bits at
  `[x0,#0x228]` using masks such as `0xfffffe3f` and values `0x40`, `0x80`,
  and `0x100`.
- `SetKernelFmt(ZinHWKernelFmt)` writes low 2 bits at `[x0,#0x4d4]`.
- `SetNEBinaryPoint(int)` writes a 6-bit field into `[x0,#0x4d8]`:

  ```asm
  ldr w8, [x0, #0x4d8]
  bfi w8, w1, #8, #6
  str w8, [x0, #0x4d8]
  ret
  ```

- `SetNEPostScale(optional<ZinKernelComponentInfo>)` converts a scalar through
  `fcvt h0, s0` before writing `[x0,#0x4d8]` and `[x0,#0x4e4]`.
- `SetNEBias(optional<ZinKernelComponentInfo>)` also converts through
  `fcvt h0, s0` before writing `[x0,#0x4d8]` and `[x0,#0x4e0]`.

Relevant compiler strings for v20 include:

- `hw.ne_config.ane_ne_config.mac_cfg.op_mode == ane_ne_mac_cfg_op_mode_conv_v20`
- `hw.ne_config.ane_ne_config.mac_cfg.kernel_mode == ane_ne_mac_cfg_kernel_mode_kernel_v20`
- `hw.ne_config.ane_ne_config.mac_cfg.binary_point == 0 || (...)`
- `hw.common_config.ane_common_config.ch_cfg.in_fmt == ZinHWTraits<20>::ane_common_ch_cfg_in_fmt_fp16`
- `in_fmt != ZinHWTraits<20>::ane_common_ch_cfg_in_fmt_bf16`
- `hw.ne_config.ane_ne_config.kernel_cfg.kernel_fmt != ZinHWTraits<20>::ane_ne_kernel_cfg_kernel_fmt_fp16`
- `hw.ne_config.ane_ne_config.kernel_cfg.sparse_fmt == 0 || (!double_int8 && in_fmt != ane_common_ch_cfg_in_fmt_fp16_v20) || ...`
- `2xInt8 mode is not supported`
- `Accumulator Retention failed.`

Driver/HAL evidence from this machine:

- `ioreg -l -r -c ANEDriver` shows `ANEDriverRoot` as class
  `H1xANELoadBalancer`, with `CFBundleIdentifier =
  com.apple.driver.AppleH16ANEInterface`. It also shows a user client created
  by `aned` and multiple `H1xANELoadBalancerDirectPathClient` instances.
- `ioreg -l -r -c ANEHWDevice` shows `H11ANE` as class `H11ANEIn`, matched by
  `IONameMatch = ane,t8020`, with `FirmwareLoaded = Yes` and device properties:
  `ANEDevicePropertyANEVersion = 192`,
  `ANEDevicePropertyANEMinorVersion = 17`,
  `ANEDevicePropertyNumANECores = 16`, and
  `ANEDevicePropertyTypeANEArchitectureTypeStr = h16g`.
- `kmutil showloaded` lists `com.apple.driver.AppleH16ANEInterface (9.511.3)`
  and `com.apple.driver.AppleT8132ANEHAL (9.511.3)` as loaded.
- The on-disk kext directories under `/System/Library/Extensions` contain
  Info.plist personalities, while their executable code is in kernel
  collections. The visible `/System/Library/KernelCollections/*.kc` files in
  this environment are reported by `file` as `x86_64`, so they were not used as
  arm64e ANE driver disassembly evidence in this document.

Current boundary:

- This is strong evidence that the private benchmark path reaches native ANE
  compiler/runtime/driver machinery and that FP16/INT8 conv is lowered to an NE
  TD programming path, not to a CPU/GPU fallback in this backend.
- It does not prove the physical silicon MAC array implementation or the exact
  accumulator datatype. In particular, the FP16 encoding of TD postscale/bias
  proves the descriptor representation for those fields, not that the internal
  MAC accumulation is FP16, FP32, fixed-point, or some proprietary mixed format.

## NPU-only Evidence

Evidence used by the benchmark:

- The model is submitted directly through `_ANEInMemoryModel evaluateWithQoS`.
- It does not use Core ML compute-unit hints, so there is no Core ML CPU/GPU
  fallback path in this backend.
- `powermetrics --samplers cpu_power -f plist` exposes `ane_power`, `gpu_power`,
  and `cpu_power`. NPU stress mode samples this while ANE eval is running.
- On this MacBook Air M4, INT8 stress reached about `35 TOPS` with `ane_power`
  around `1.6W` and `gpu_power` around `0.3W`.
- NPU stress with `--power powermetrics` also samples
  `powermetrics --samplers tasks --show-process-gpu` concurrently and reports
  `appleflops_gpu_ms_s=0.00` in the output row. A manual INT8 NPU stress check
  showed the same `GPU ms/s=0.00` while sustaining about `35.2 TOPS`.

Limits:

- Apple does not expose an official ANE occupancy counter through public APIs.
- `util_ref_m4=100%` means throughput is at the `maderix/ANE` M4 reference peak
  for this workload, not an official hardware occupancy value.

## BF16 Probe

BF16 is not reported as a valid NPU score on this machine. The ANE compiler
returns `InvalidMILProgram`.

Tried variants:

| Variant | Target | Result |
| --- | --- | --- |
| `true_bf16` | `ios18` | `InvalidMILProgram` |
| `true_bf16` | `ios19` | `InvalidMILProgram` |
| `bf16_io_fp16` | `ios18` | `InvalidMILProgram` |
| `bf16_io_fp16` | `ios19` | `InvalidMILProgram` |
| `fp32_bf16_fp16` | `ios18` | `InvalidMILProgram` |
| `fp32_bf16_fp16` | `ios19` | `InvalidMILProgram` |

Probe file:

- `_probe_ane_bf16_variants.m`

## FP32 Probe

FP32 is not reported as an NPU score. The only FP32-shaped path that compiles is
`fp32_io_fp16_compute`: FP32 live-in/live-out tensors are explicitly cast to and
from FP16, while the conv chain itself uses FP16 tensors and FP16 weights. The
main benchmark disables NPU FP32 to avoid reporting this as FP32 ANE compute.

Local probe results on this MacBook Air M4:

| Variant | Target | Result |
| --- | --- | --- |
| `true_fp32` | `ios18` | `ANECCompile() FAILED` |
| `true_fp32` | `ios19` | `ANECCompile() FAILED` |
| `fp32_io_fp16_compute` | `ios18` | compile/load/evaluate OK |
| `fp32_io_fp16_compute` | `ios19` | compile/load/evaluate OK |
| `fp32_input_fp16_weight_no_cast` | `ios18` | `ANECCompile() FAILED` |
| `fp32_input_fp16_weight_no_cast` | `ios19` | `ANECCompile() FAILED` |

The unified log around the failed `true_fp32` compile contains:

- `Error: ANE cannot handle intermediate tensor type <private>`
- `Function call to BuildLayerGraph() failed ... ZinCompilerCoreClassic.cpp:302`

Static dyld shared-cache extraction found additional compiler/runtime strings:

- `Error: OpLayer %s input format not acceptable (UINT8/SINT8/FP16)`
- `Error: OpLayer %s output format not acceptable (UINT8/SINT8/FP16)`
- `Incompatible element type for ANE: expected fp16, si8, or ui8`
- `Conv input, weights, and output must be supported dtype.`
- `Fp32 precision (for bias) not supported by ANE.`
- `kANE_FP16_CYCLES:` and `kANE_INT8_CYCLES` exist in
  `AppleNeuralEngine.framework`; no matching `FP32_CYCLES` string was found.

This does not prove the physical silicon lacks every FP32-capable block. It does
prove that the local ANE compiler/runtime path we can reach does not accept a
native FP32 conv compute graph, while it accepts the FP32-IO/FP16-compute graph.
The benchmark therefore reports only NPU FP16 and INT8 peak scores by default.

Probe files:

- `_probe_ane_fp32_variants.m`
- `_extract_dsc.mm`

## Performance Stats Probe

Runtime reflection shows:

- `_ANEPerformanceStats`
- `_ANEPerformanceStatsIOSurface`
- `_ANEInMemoryModel perfStatsMask`
- `_ANERequest perfStatsArray`

But direct `_ANEPerformanceStats` objects are not accepted by `_ANERequest`;
the request expects objects with `statType`. `_ANEPerformanceStatsIOSurface`
objects are accepted, but the stats IOSurface remained all-zero in local probes,
and `request.perfStats` stayed `nil`.

Current conclusion: `powermetrics` rail sampling plus near-reference throughput
is stronger local evidence than `_ANEPerformanceStats` for this benchmark.

Probe file:

- `_probe_ane_perfstats.m`
