# MTFLOPS

Mac Apple Silicon 原生算力/能效测试工具。

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

更多参数与实现细节请直接阅读 `main.cpp`、`gpu_bench.mm`、`shaders/gemm.metal`。
