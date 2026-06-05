# AppleFLOPS

Apple Silicon / Apple 设备原生算力测试工具，支持 macOS CLI 和 iOS runner，覆盖 CPU/GPU 多精度跑分，并提供 NPU(ANE private) 的 FP16/INT8 peak 跑分与 BF16 probe。

## 下载（推荐）
直接从 Release 页面下载预编译包 `AppleFLOPS-macos-arm64.zip`：
```bash
curl -L -o AppleFLOPS-macos-arm64.zip https://github.com/baicai-1145/AppleFLOPS/releases/download/auto-2-0623603/AppleFLOPS-macos-arm64.zip
unzip AppleFLOPS-macos-arm64.zip
cd AppleFLOPS-macos-arm64
```

如果从源码仓库 `git clone`，则需要先运行 `make` 自行编译。

## 推荐一键跑分
一次跑完当前可信支持矩阵：CPU FP32/FP16/BF16/INT8、GPU FP32/FP16/BF16、NPU FP16/INT8：
```bash
./appleflops --unit all --precision all --n 1024 --warmup 1 --repeats 3
```
其中 NPU 固定使用 `N=512` channels；GPU INT8 不在 `--unit all` 默认矩阵内，因为当前公开路径不是同口径的矩阵单元峰值。

## 实测数据

下表是本项目当前记录的本机实测最高点，用于给读者一个量级参考，不代表所有机器、系统版本、散热状态下都能复现。浮点单位是 TFLOPS，INT8 单位是 TOPS。

MacBook Air M4（16GB，macOS，本机记录于 2026-06-05）：

| Unit | FP32 | FP16 | BF16 | INT8 |
|---|---:|---:|---:|---:|
| CPU SME | 2.320 | 2.310 | 2.324 | 4.677 |
| GPU Metal simdgroup | 2.783 | 2.913 | 2.889 | 不记录 |
| NPU / ANE private | 不记录 | 18.256 | 不记录 | 34.558 |

iPhone 17 / A19（iPhone18,3，iOS 26.5.1，本机记录于 2026-06-06）：

| Unit | FP32 | FP16 | BF16 | INT8 |
|---|---:|---:|---:|---:|
| CPU SME | 1.394 | 1.395 | 1.394 | 4.697 |
| GPU Metal simdgroup | 1.880 | 1.885 | 1.885 | 不记录 |
| NPU / ANE private | 不记录 | 17.061 | 不记录 | 32.091 |

口径说明：
- CPU 使用 SME MOPA peak workload；FP16/BF16 是 FP16/BF16 输入、FP32 accumulate，INT8 是 INT32 accumulate。
- GPU FP32/FP16/BF16 使用 Metal `simdgroup_matrix` peak workload；GPU INT8 不记录可信矩阵单元峰值。
- NPU 使用 private ANE conv1x1-chain workload；FP32/BF16 当前不记录为 native ANE 分数。

带全程功耗采样时需要 sudo；每一行 benchmark 会在 `powermetrics` 完整采样窗口内持续循环对应 workload，并从 `cpu_power` plist 同时读取 CPU/GPU/ANE 三条 rail。`Watts` 列按当前行所属硬件选择对应 rail 的 median，Note 中会显示 `median(cpu/gpu/ane)=...`、`avg`、`range` 和 `samples`：
```bash
sudo ./appleflops --unit all --precision all --n 1024 --warmup 1 --repeats 3 --power powermetrics --verify 0
```

如果想把 CPU/GPU/NPU 同时压起来，当作整机压力测试，用 `--unit all --stress`。开启功耗采样时，每轮会在 `--power-window-ms * --power-samples` 的完整采样期间持续循环 CPU、GPU、NPU 三个 workload，并同时采样三条功耗 rail：
```bash
sudo ./appleflops --unit all --precision fp16 --stress 10 --n 2048 --gpu-workload peak --gpu-inner 512 --gpu-batch 8 --warmup 0 --repeats 1 --power powermetrics --power-every 1 --verify 0
```
更稳定的功耗口径建议把采样窗口拉长并取多样本 median，例如：
```bash
sudo ./appleflops --unit all --precision fp16 --stress 5 --n 2048 --gpu-workload peak --gpu-inner 512 --gpu-batch 8 --warmup 0 --repeats 1 --power powermetrics --power-window-ms 3000 --power-samples 5 --power-every 1 --verify 0
```
`Watts` 列使用对应 rail 的 median；Note 中的 `range` 用于判断采样波动是否仍然过大。若 `range` 仍很宽，说明系统调度、DVFS、后台任务或被动散热状态仍在影响功耗口径。

`--unit all --stress --power powermetrics` 默认会把 CPU SME worker 限到硬件线程数，避免默认 `APPLEFLOPS_CPU_THREADS=硬件线程数*6` 的 CPU 峰值探针挤占 ANE/GPU 提交线程和 `powermetrics` 采样线程。若你要纯粹最大 CPU 压力，可以显式设置 `APPLEFLOPS_CPU_THREADS=60`，但 ANE/GPU rail 的 `range` 可能会明显变宽，甚至短窗口内采到 0。

## GPU + 功耗 + 热降频提示
`powermetrics` 需要 sudo：
```bash
sudo ./appleflops --unit gpu --precision all --gpu-workload gemm --kernel v4 --n 1024 --warmup 2 --repeats 5 --gpu-batch 32 --gpu-inner 16 --stress 10 --power powermetrics --power-every 2
```

不测功耗时，可以先跑浮点 GPU 快速分数：
```bash
./appleflops --unit gpu --precision fp16 --gpu-workload gemm --kernel v4 --n 1024 --warmup 2 --repeats 5 --gpu-batch 32 --gpu-inner 16 --gpu-storage private
```

## CPU/GPU 多精度 smoke test
```bash
./appleflops --unit cpu --precision all --n 1024 --warmup 1 --repeats 3
./appleflops --unit gpu --precision all --gpu-workload gemm --kernel v4 --n 128 --warmup 0 --repeats 1
```
CPU 路径现在是 SME MOPA peak probe，`N` 控制 inner loop 规模，不再表示 GEMM 矩阵尺寸。`APPLEFLOPS_CPU_THREADS` 可覆盖 CPU worker 数量；默认使用多于物理核心的 worker 来减少调度空洞。

CPU/GPU 利用率采样示例：
```bash
./appleflops --unit cpu --precision fp32 --stress 14 --n 4096 --warmup 0 --repeats 1 --verify 0
./appleflops --unit gpu --precision bf16 --gpu-workload peak --n 2048 --gpu-inner 512 --gpu-batch 8 --stress 6 --warmup 0 --repeats 1 --verify 0
```
本机 MacBook Air M4 上，`powermetrics` 采样到 CPU `E-Cluster HW active residency=100.00%`、`P-Cluster HW active residency=100.00%`、两者 `idle residency=0.00%`；GPU 长 peak command 采样到 `GPU HW active residency=100.00%`、`GPU idle residency=0.00%`。这能证明 CPU cluster 和 GPU 在测速窗口内处于硬件 active 满载状态；进程级 `CPU ms/s` 和单个 CPU core residency 仍会受调度、采样与系统任务影响，不能把它解释成每个逻辑核逐项都是 100%。

## NPU / ANE private 跑分
NPU 路径基于 `maderix/ANE` 的 private `_ANEInMemoryModel` 思路，使用 ANE conv1x1-chain peak workload。`N` 表示 channels，`--npu-spatial` 和 `--npu-depth` 控制空间大小与 conv 层数：
```bash
./appleflops --unit npu --precision fp16 --n 512 --npu-spatial 64 --npu-depth 128 --warmup 5 --repeats 5
./appleflops --unit npu --precision int8 --n 512 --npu-spatial 64 --npu-depth 128 --warmup 5 --repeats 5
```
带 ANE rail 采样的压力测试需要 sudo；NPU stress 模式会让 `powermetrics` 采样窗口和 ANE eval 重叠，并从 plist 的 `ane_power` 字段读取 NPU 功耗：
```bash
sudo ./appleflops --unit npu --precision fp16 --stress 2 --power powermetrics --power-every 1 --warmup 2 --repeats 80 --verify 0
sudo ./appleflops --unit npu --precision int8 --stress 2 --power powermetrics --power-every 1 --warmup 2 --repeats 80 --verify 0
```
MacBook Air M4 和 iPhone 17 / A19 的代表性实测结果见上方“实测数据”表。
带 power 的 NPU stress 行会额外打印类似 `powermetrics(cpu_power plist rails interval=1000ms samples=3 median) | median(cpu/gpu/ane)=...W | ... | powermetrics(tasks) appleflops_gpu_ms_s=0.00`。这能证明 ANE dedicated rail 在工作，同时 `appleflops` 进程没有 GPU time；它不是 ANE 硬件占用率 counter。

注意：
- NPU 后端使用 Apple 私有 API：`AppleNeuralEngine.framework` / `_ANEInMemoryModel`，只适合研究用途。
- FP16 和 INT8 W8A8 是当前稳定 ANE peak 路径。
- FP32 不再报告 NPU 分数：本机 `true_fp32` ANE conv 编译失败；过去的 `fp32_io_fp16_compute` 只是 FP32 I/O + FP16 compute，不是原生 FP32 ANE 算力。
- BF16 路径目前会尝试 `bf16_io_fp16_compute` probe；本机 ANE compiler 仍返回 `InvalidMILProgram`。`true_bf16`、`bf16_io_fp16`、`fp32_bf16_fp16` 在 `ios18/ios19` 目标下也都失败，因此不会伪报 BF16 NPU 分数。

浮点精度输出 TFLOPS；INT8 输出 TOPS。CPU 使用 SME MOPA peak probe；GPU INT8 GEMM 使用 MPS QMM；NPU 使用 private ANE conv1x1-chain。

实现口径：
- CPU FP32/FP16/BF16/INT8 主测速全部使用 Arm SME MOPA peak probe，分别走 `svmopa_za32_f32/f16/bf16/s8`，FP16/BF16 输出 FP32 accumulation，INT8 输出 INT32 accumulation。
- 旧 CPU GEMM / Accelerate / NEON / I8MM 代码保留为历史对照实现，不再作为 CPU 主测速路径。
- GPU FP32/FP16/BF16 使用 Metal `simdgroup_matrix`。
- GPU INT8 GEMM 默认使用 Metal-backed MPS `MPSNDArrayQuantizedMatrixMultiplication`，输出 INT32；它是公开框架里的量化 GEMM 路径，不等同于可手写调用的 INT8 matrix-unit kernel，`--gpu-batch/--gpu-inner` 对该路径会被忽略。INT8 peak probe 仍是 scalar Metal。当前 Metal `simdgroup_matrix` 不接受 int8/int32 元素类型，Metal 4 cooperative tensor 头文件也没有公开的 GEMM/MMA 构造入口。
- NPU FP16/INT8 使用 private ANE `_ANEInMemoryModel` 直接 evaluate；FP32 不再测速；BF16 probe 当前编译失败。输出 `npu_only=private_ane_eval` 表示没有走 Core ML CPU/GPU fallback。`util_ref_m4` 是相对 maderix M4 参考峰值的估算；接近 100% 代表吞吐接近该 workload 的 M4 参考峰值，不是 Apple 官方暴露的 ANE occupancy counter。`--power powermetrics` 会读取 `ane_power` rail 作为额外证据。

M4 GPU INT8 / MXU 的逆向探索过程见 [docs/m4-gpu-int8-investigation.md](docs/m4-gpu-int8-investigation.md)。

NPU/ANE private API 的 BF16 与 perfStats 探索过程见 [docs/npu-ane-private-investigation.md](docs/npu-ane-private-investigation.md)。

第三方许可说明见 [docs/third-party-licenses.md](docs/third-party-licenses.md)。

更多参数与实现细节请直接阅读 `main.cpp`、`gpu_bench.mm`、`shaders/gemm.metal`。
