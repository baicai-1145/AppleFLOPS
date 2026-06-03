# MTFLOPS

Mac Apple Silicon 原生算力/能效测试工具，支持 CPU/GPU 的 FP32、FP16、BF16、INT8 GEMM 跑分。

## 下载（推荐）
下载 `https://github.com/baicai-1145/MTFLOPS/releases/download/v1.0.0/MTFLOPS.zip` 并解压：
```bash
curl -L -o MTFLOPS.zip https://github.com/baicai-1145/MTFLOPS/releases/download/v1.0.0/MTFLOPS.zip
unzip MTFLOPS.zip
cd MTFLOPS
```

## 推荐一键跑分（GPU + 功耗 + 热降频提示）
`powermetrics` 需要 sudo：
```bash
sudo ./mtflops --unit gpu --precision all --gpu-workload gemm --kernel v4 --n 1024 --warmup 2 --repeats 5 --gpu-batch 32 --gpu-inner 16 --stress 10 --power powermetrics --power-every 2
```

不测功耗时，可以先跑浮点 GPU 快速分数：
```bash
./mtflops --unit gpu --precision fp16 --gpu-workload gemm --kernel v4 --n 1024 --warmup 2 --repeats 5 --gpu-batch 32 --gpu-inner 16 --gpu-storage private
```

## CPU/GPU 多精度 smoke test
```bash
./mtflops --unit cpu --precision all --n 128 --warmup 0 --repeats 1
./mtflops --unit gpu --precision all --gpu-workload gemm --kernel v4 --n 128 --warmup 0 --repeats 1
```
`N=128` smoke test 只用于确认各精度路径能运行，不代表 CPU/GPU 峰值算力。

浮点精度输出 TFLOPS；INT8 输出 TOPS。

实现口径：
- CPU FP32 使用 Accelerate `cblas_sgemm`。
- CPU FP16/INT8 在支持对应特性的机器上使用 NEON 快路径；CPU BF16 目前使用 verified typed kernel。
- GPU FP32/FP16/BF16 使用 Metal `simdgroup_matrix`；GPU INT8 使用 Metal scalar GEMM/probe，因为当前 Metal `simdgroup_matrix` 不接受 int8/int32 元素类型。

更多参数与实现细节请直接阅读 `main.cpp`、`gpu_bench.mm`、`shaders/gemm.metal`。
