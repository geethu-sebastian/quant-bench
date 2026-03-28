| Method            | Precision      |   Model Size (MB) | Compression   | Throughput (tok/s)   |   TTFT (ms) |   Per-Token (ms) |   Peak Mem (MB) |   Perplexity |
|-------------------|----------------|-------------------|---------------|----------------------|-------------|------------------|-----------------|--------------|
| FP32 Baseline     | FP32           |             486.7 | 1.00x         | 24.8 ± 0.3           |        49.1 |             40.4 |            1029 |        25.9  |
| Dynamic INT8      | INT8 (dynamic) |             486.7 | 1.00x         | 37.9 ± 1.3           |        34.5 |             26.4 |            1933 |     11688.2  |
| Static PTQ INT8   | INT8 (static)  |             486.7 | 1.00x         | 37.1 ± 1.1           |        52.3 |             27   |            3439 |     11688.2  |
| ONNX Runtime INT8 | INT8 (ONNX)    |             120.1 | 4.05x         | 23.1 ± 1.1           |         0   |             43.3 |            3988 |        40.12 |