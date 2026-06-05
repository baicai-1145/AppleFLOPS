# M4 GPU INT8 / MXU 探索记录

本文记录一次在 MacBook Air M4 上排查 GPU INT8 GEMM 性能偏低的过程。重点结论是：**当前 M4 + 当前 Metal/MPS runtime 下，没有可创建并 dispatch 的 GPU INT8 matrix-unit/MXU 路径**。这不等价于证明硅片物理上绝对没有该单元；它证明的是当前软件栈不暴露、不可达。

## 环境

- 机器：MacBook Air M4
- 系统：macOS 26.5，arm64
- GPU feature：
  - `supportsFamily:MTLGPUFamilyApple9 == true`
  - `supportsFamily:MTLGPUFamilyApple10 == false`

## 问题背景

一开始 GPU INT8 跑分明显低于“TOPS”直觉。需要区分三种完全不同的概念：

- 普通 shader SIMD/ALU 的 INT8：GPU 可以做通用整数运算，但这不是高吞吐矩阵单元。
- MPS public quantized GEMM：`MPSNDArrayQuantizedMatrixMultiplication` 是公开 API，但不保证使用 INT8 MXU。
- INT8 matrix-unit/MXU：例如 MPS 私有 `gemm_i2i8_a18` 中的 `simdgroup_matrix<T,16,16x16>` 变体，这才是高 TOPS 的关键路径。

## Public Metal / AIR 先验

手写 AIR 调用 INT8 widening matrix intrinsic 时，可以 assemble/package，但在 M4 上创建 pipeline 失败：

```text
simdgroup_matrix<T,16,16x16> operations are supported by GPUFamily10 and later
```

这说明 public Metal 编译器对该 INT8 matrix 形态有 Family10 gate。M4 是 Apple GPU Family9，因此 public Metal 路径不能直接创建这个 INT8 matrix pipeline。

## MPS 私有 metallib 里确实有 INT8 A18 kernel

MPSNDArray 私有 metallib 中能看到相关函数名：

```text
gemm_i2i8_a18
q4q8_gemm
conv2d_a18_int32_int8_int8
```

`gemm_i2i8_a18` 存在并不代表 M4 能执行它。它还要经过 MPS function constants、Metal specialized function、pipeline creation 和 device feature validation。

## Public QMM 默认没有走 A18

测试探针：`_probe_qmm_dispatch.mm`

典型编译命令：

```bash
xcrun clang++ -std=c++17 -ObjC++ -fobjc-arc _probe_qmm_dispatch.mm \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -o /tmp/probe_qmm_dispatch
```

典型输入：

```bash
/tmp/probe_qmm_dispatch rank3-nt-sr3-sf16-oi32 1024 1
```

其中：

- A/B：rank 3，INT8
- B：transpose view，即 NT
- scale：rank 3，FP16
- 输出：INT32

`xctrace` 的 Metal shader list 显示 public QMM 实际 dispatch 的是：

```text
ndArrayLUTGEMV_xReduce
```

没有看到：

```text
gemm_i2i8_a18
q4q8_gemm
```

## MPSNDArray 反汇编定位

从 dyld shared cache 提取的 MPSNDArray 二进制：

```text
/Volumes/2T/dsc_arm64e_extract/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks/MPSNDArray.framework/Versions/A/MPSNDArray
```

本次进程中 MPSNDArray slide：

```text
0x96b0000
```

关键符号：

```text
file 0x18faff82c -> runtime 0x1991af82c
EncodeQuantizedMatrixMultiplication

file 0x18faa851c -> runtime 0x19915851c
MPSNDArrayMatMulA18DeviceBehavior::IsInt8AffineSupportedQuantization

file 0x18faa8bb8 -> runtime 0x199158bb8
MPSNDArrayMatMulA18DeviceBehavior::EncodeQuantizedMatrixMultiplicationInt8Affine

file 0x18fafc9bc -> runtime 0x1991ac9bc
MPSNDArrayMatMulDeviceBehavior::IsInt8AffineSupportedQuantization

file 0x18fb009b0 -> runtime 0x1991b09b0
LUTGEMV decision path
```

`EncodeQuantizedMatrixMultiplication` 中的关键分支：

```text
0x1991b0818  调 vtable #0x20：Int8Affine support
0x1991b0844  检查 w0，false 则跳过 Int8Affine encode
0x1991b09b0  进入 LUTGEMV decision path
```

LLDB 断点显示 public QMM 默认路径是：

```text
EncodeQuantizedMatrixMultiplication
-> MPSNDArrayMatMulDeviceBehavior::IsInt8AffineSupportedQuantization
-> w0 = 0
-> LUTGEMV path
```

没有命中 A18 support：

```text
MPSNDArrayMatMulA18DeviceBehavior::IsInt8AffineSupportedQuantization
```

## 为什么没有命中 A18：MPS 选择了 A14 behavior

在 behavior factory 上下断点：

```text
0x199157968  ndArrayA14MatMulDeviceBehavior
0x199157008  ndArrayA18MatMulDeviceBehavior
```

public QMM 初始化时命中：

```text
ndArrayA14MatMulDeviceBehavior
```

调用栈：

```text
MPSCore`MPSLibrary::MPSLibrary
MPSCore`MPSDevice::GetMPSLibrary_DoNotUse
MPSNDArray`-[MPSNDArrayMultiaryBase initWithDevice:sourceCount:]
MPSNDArray`-[MPSNDArrayMultiaryKernel initWithDevice:sourceCount:]
MPSNDArray`-[MPSNDArrayMatrixMultiplication initWithDevice:sourceCount:]
MPSNDArray`-[MPSNDArrayQuantizedMatrixMultiplication initWithDevice:...]
```

也就是说：M4 上 public QMM 继承的 matmul behavior 是 A14，不是 A18。A14 behavior 没有 override Int8Affine support/encode，因此虚表 slot 落到基类实现，返回 false。

`MPSNDARRAY_FORCE_MXU=1` 也没有改变这一点。断点仍然显示：

```text
MPSNDArrayMatMulDeviceBehavior::IsInt8AffineSupportedQuantization
w0 = 0
-> LUTGEMV
```

## 强制 A18 实验

为了区分“输入条件不满足”还是“MPS 根本没选 A18”，用 LLDB 在 A14 factory 入口把 PC 跳到 A18 factory：

```lldb
breakpoint set -a 0x199157968
continue
thread jump --address 0x199157008
continue
```

强制后结果：

```text
命中 MPSNDArrayMatMulA18DeviceBehavior::IsInt8AffineSupportedQuantization
w0 = 1

命中 MPSNDArrayMatMulA18DeviceBehavior::EncodeQuantizedMatrixMultiplicationInt8Affine
```

这说明当前输入参数可以满足 A18 Int8Affine support。默认不走 A18 的主要原因是 MPS behavior 选择，不是参数完全不合格。

但继续运行时，在创建 A18 matrix pipeline 时失败：

```text
MPSKernel MTLComputePipelineStateCache unable to load function gemm_a18.
simdgroup_matrix<T,16,16x16> operations are supported by GPUFamily10 and later
```

`of32` 和 `oi32` 输出变体都遇到同类 Family10 gate。

## 私有 metallib 直接 pipeline 验证

额外写过临时探针，直接加载：

```text
/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks/MPSNDArray.framework/Versions/A/Resources/default.metallib
```

结果：

```text
device=Apple M4 supportsApple9=1 supportsApple10=0

gemm_i2i8_a18:
  index=115 type=85 required=1 name=fc0
  index=122 type=33 required=1 name=MPSNDArrayRefSize
```

`gemm_i2i8_a18` 的空/全零 function constants 可以建 pipeline，但那不是实际 MXU 变体。对 `fc0` 做 fuzz：

```text
fc0 byte0=0xff:
pipeline gemm_i2i8_a18: fail
error=simdgroup_matrix<T,16,16x16> operations are supported by GPUFamily10 and later
```

进一步扫 byte0 的 0..255 发现：当 packed 配置进入某些 16x16 matrix 变体时，`gemm_i2i8_a18` 会触发同一个 Family10 gate。全零能建 pipeline，只能说明它没有选择 INT8 matrix-unit 变体，不能说明 M4 能跑 INT8 MXU。

## 能不能绕过 Family10 gate

已经绕过了第一道门：

```text
MPS A14 behavior selection
```

但第二道门是 Metal pipeline compiler / driver feature validation：

```text
simdgroup_matrix<T,16,16x16> operations are supported by GPUFamily10 and later
```

这不是 `MPSKernelOptionsSkipAPIValidation` 能跳过的 MPS API validation。只 hook `supportsFamily:` 也大概率不够，因为 Metal compiler/driver 有底层 device feature mask。

理论上继续往下要 patch Metal compiler、AGX driver 或 GPU firmware，或者拿 Family10 设备编出最终 GPU ISA 后尝试注入。但 Metal 没有公开 raw GPU ISA pipeline 入口，且还有签名、验证和 firmware 风险。因此这条路不适合作为本工具的可用实现路径。

## ANE 能否帮助判断 GPU INT8

ANE 不能直接证明 GPU 有没有 INT8 matrix-unit。ANE 和 GPU 是不同执行后端：

```text
ANE private compiler/runtime -> AppleNeuralEngine.framework / ANE rail
GPU compiler/runtime         -> Metal / MPS / AGX driver / GPU rail
```

本机 ANE 证据只能证明 ANE 自己接受 FP16/INT8、拒绝当前可达的 native FP32/BF16 conv 变体；它不能推出 GPU 上存在或不存在 INT8 MXU。

不过 ANE 逆向方法可以复用到 CoreML/MPSGraph 的 GPU 路径上：

```text
dyld shared-cache 提取
private framework strings/symbols
unified log
xctrace Metal shader list
powermetrics rail separation
LLDB 断点/强制 behavior
```

补查 CoreML/MPSGraph/MPSNDArray 本地二进制后，看到的关系是：

```text
MPSGraph 有 ANE_region / mpsx.ane / mpsx.gpu / GPURegionCallOpHandler。
MPSGraph 有 QuantizedMatMul / QuantizedMatMulFusionOpHandler / GPUQuantizationOps。
MPSGraph 有 MPSGRAPH_DISABLE_GPU_QUANT_OPS / enableGPUQuantOps / disableGPUQuantOps。
CoreML 有 MLComputeUnitsCPUAndGPU / MLComputeUnitsCPUAndNeuralEngine / MLComputeUnitsAll。
```

这说明 CoreML/MPSGraph 会做 ANE/GPU placement，也有 GPU 量化 op lowering。但这条 lowering 最终仍要落到 MPS/Metal GPU kernel。当前已经观察到的 GPU 侧结果仍是：

```text
public QMM 默认走 A14/base -> LUTGEMV。
强制 A18 behavior 后，可以进入 Int8Affine support/encode。
真正 A18 matrix pipeline 创建时被 GPUFamily10 gate 拦住。
```

所以通过 ANE 能做的是“排除混跑/误归因”和“复用逆向手段”：

```text
CoreML/MPSGraph GPU-only + ANE disabled -> 看 shader list 是否出现 gemm_i2i8_a18。
CoreML/MPSGraph all/ANE placement       -> 用 ane_power/gpu_power 区分实际后端。
MPSGRAPH_DISABLE_GPU_QUANT_OPS 对照     -> 看量化 matmul 是否退化或改路由。
```

如果 GPU-only 量化 workload 仍只出现 LUTGEMV/scalar，或强制 A18 后仍触发 Family10 gate，那么它只能加强“当前软件栈不可达”的结论；不能把 ANE 的 INT8 成功当成 GPU INT8 MXU 存在的证明。

## MPSGraph 量化 matmul 补查

为了排除“public MPSNDArray QMM 不走 A18，但 MPSGraph 另有 GPU INT8 路径”的可能，补写了两个探针：

```text
_probe_mpsgraph_quant_matmul.mm
_probe_mpsgraph_mlir_source.mm
```

public MPSGraph 的直接 INT8 matmul 不成立：

```bash
/tmp/probe_mpsgraph_quant_matmul direct-i8 128 1
```

报错：

```text
'mps.matmul' op operand #0 must be tensor of floating point values or tensor of complex values, but got 'tensor<128x128xsi8>'
failed assertion `original module failed verification'
```

也就是说，public `matrixMultiplication` 不接受 raw INT8 tensor。

public dequant + matmul 可以运行，但 `getIR` 是显式反量化再做浮点 matmul：

```mlir
%3 = "mps.dequantize"(%arg0, %0, %1, %2) <{dtype = f16}>
%4 = "mps.dequantize"(%arg1, %0, %1, %2) <{dtype = f16}>
%5 = "mps.matmul"(%3, %4)
```

Metal System Trace 中看到的 shader 是：

```text
mmul_kernel_a16_half_half_float
```

没有看到：

```text
gemm_i2i8_a18
NDArrayQuantizedMatmul_int8Affine
mpsx.quantized_matmul
```

这说明 public MPSGraph dequant+matmul 当前没有 fusion 到 GPU INT8 quantized matmul。

随后用 MPSGraph 私有入口直接加载 MLIR source：

```objc
-[MPSGraphExecutable initWithMLIRSourceFromURL:executableDescriptor:]
```

先验证普通 `mps.dequantize + mps.matmul` MLIR 能运行，再手写 `mpsx.quantized_matmul`：

```mlir
%3 = "mpsx.quantized_matmul"(%arg0, %0, %1, %2, %arg1, %0, %1, %2) {
  input_quant_params_axis = 0 : si32,
  operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>,
  output_type = f16,
  transpose_lhs = false,
  transpose_rhs = false,
  weights_quant_params_axis = 0 : si32
} : (tensor<128x128xsi8>, tensor<f32>, tensor<si8>, tensor<f32>,
     tensor<128x128xsi8>, tensor<f32>, tensor<si8>, tensor<f32>)
     -> tensor<128x128xf16>
```

关键语法/类型结论：

```text
attributes 要用普通 `{...}`，不能用 `<{...}>` properties。
axis attr 要用 signed integer：`0 : si32`。
zeroPoint 类型要和 input/weights 一致：INT8 输入对应 `tensor<si8>`，不是 public dequant 的 `tensor<si32>`。
```

这个 private op 可以执行，且 `getIR` 确认 executable 中仍保留 `mpsx.quantized_matmul`，不是提前改写成普通 matmul。

LLDB 断点显示 private `mpsx.quantized_matmul` 的 GPU 执行路径是：

```text
MPSGraphExecutable runWithMTLCommandQueue
-> GPURegionRuntime::evaluateOps
-> GPU::QuantizedMatMulOpHandler::encodeNDArrayOp
-> -[MPSNDArrayMultiaryKernel encodeToMPSCommandEncoder:...]
-> EncodeQuantizedMatrixMultiplication
-> MPSNDArrayMatMulDeviceBehavior::IsInt8AffineSupportedQuantization
```

其中 base behavior 的 support 返回值是：

```text
w0 = 0
```

A18 behavior 断点没有命中：

```text
MPSNDArrayMatMulA18DeviceBehavior::IsInt8AffineSupportedQuantization hit count = 0
```

继续运行后命中 fallback：

```text
EncodeArrayLUTGEMV
-> EncodeQuantizedMatrixMultiplication
-> -[MPSNDArrayMultiaryKernel encodeToMPSCommandEncoder:...]
-> GPU::QuantizedMatMulOpHandler::encodeNDArrayOp
```

所以，手写 private `mpsx.quantized_matmul` 确实能到 MPSGraph GPU quant handler，但在 M4 上仍下沉到 MPSNDArray QMM 的 base behavior，并回落到 LUTGEMV；它没有默认进入 A18 INT8 affine matrix 路径。

`MPSGRAPH_DISABLE_GPU_QUANT_OPS=1` 对这个 private MLIR source 入口也没有阻止执行，只打印：

```text
MPSGRAPH_DISABLE_GPU_QUANT_OPS EV is set.
```

这说明该环境变量不足以作为“是否走 GPU INT8 MXU”的判据，最终仍要看 LLDB 调用栈、shader list 或 pipeline creation gate。

## 结论边界

已经证明：

```text
M4 GPU 当前 Metal/MPS runtime 不暴露可用的 INT8 MXU 路径。
public QMM 默认走 A14/base -> LUTGEMV。
public MPSGraph INT8 matmul 不接受 raw INT8；dequant+matmul 走普通 FP16 matmul。
private MPSGraph `mpsx.quantized_matmul` 可执行，但默认仍走 MPSNDArray base -> LUTGEMV。
强制 A18 后，A18 support/encode 可进入，但真正 matrix pipeline 被 Family10 gate 拦住。
gemm_i2i8_a18 私有函数存在；当 function constants 选择 matrix-unit 变体时同样被 Family10 gate 拦住。
```

没有证明：

```text
M4 硅片物理上绝对没有 INT8 MXU。
```

软件层面很难证明一个单元“物理不存在”，因为“存在但被 firmware/driver/compiler 禁用”和“物理不存在”在当前观测中等价。要证明物理不存在，需要苹果微架构文档、Family10 对照设备的最终 GPU ISA 对比，或者硬件级逆向。

因此本项目的实现口径应保持为：

```text
M4 GPU INT8 只能测 public MPS QMM 或 scalar Metal shader 路径；
当前不能把结果解释为 GPU INT8 MXU/TOPS 峰值。
```
