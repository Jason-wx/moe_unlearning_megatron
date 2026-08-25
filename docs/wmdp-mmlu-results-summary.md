# WMDP / MMLU Evaluation Results

Metrics are `acc,none` from `result.json`, shown as percentages. `Delta` values are percentage-point changes relative to each model's `base`.

## Qwen

| Stage | WMDP | Bio | Chem | Cyber | MMLU | STEM | Other | Social | Humanities | Delta WMDP | Delta MMLU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 67.09 | 83.27 | 68.14 | 56.52 | 82.47 | 81.54 | 86.13 | 90.54 | 75.41 | +0.00 | +0.00 |
| fineweb_retain | 55.13 | 52.47 | 65.93 | 54.60 | 82.06 | 81.19 | 84.94 | 90.32 | 75.35 | -11.97 | -0.41 |
| wmdp_retain | 57.58 | 59.54 | 66.42 | 54.50 | 81.10 | 80.62 | 85.07 | 89.37 | 73.39 | -9.51 | -1.37 |
| sparse_experts | 56.92 | 54.12 | 67.16 | 56.62 | 82.09 | 81.00 | 84.81 | 90.67 | 75.41 | -10.17 | -0.38 |

## Intern

| Stage | WMDP | Bio | Chem | Cyber | MMLU | STEM | Other | Social | Humanities | Delta WMDP | Delta MMLU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 68.76 | 83.66 | 68.87 | 59.18 | 85.15 | 84.78 | 87.80 | 91.71 | 79.36 | +0.00 | +0.00 |
| fineweb_retain | 48.50 | 37.31 | 61.03 | 53.10 | 83.46 | 83.79 | 86.16 | 90.93 | 76.58 | -20.26 | -1.69 |
| wmdp_retain | 49.21 | 39.83 | 57.60 | 53.50 | 75.81 | 73.80 | 84.26 | 87.42 | 63.97 | -19.55 | -9.34 |
| sparse_experts | 48.75 | 38.10 | 64.46 | 52.34 | 84.46 | 84.08 | 86.00 | 91.23 | 79.28 | -20.01 | -0.69 |

## Gemma 4 26B-A4B

The RMU run uses WMDP-Bio as forget data and FineWeb as retain data. It trains all
experts and the dense MLP/shared expert in layers 7–9.

| Stage | WMDP | Bio | Chem | Cyber | MMLU | STEM | Other | Social | Humanities | Delta WMDP | Delta MMLU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 61.12 | 79.26 | 58.82 | 49.97 | 74.24 | 68.57 | 80.82 | 86.12 | 65.93 | +0.00 | +0.00 |
| rmu_bio_fineweb_retain | 55.32 | 64.81 | 56.62 | 48.97 | 74.13 | 68.47 | 80.21 | 86.51 | 65.80 | -5.81 | -0.11 |

The Bio score decreases by 14.45 percentage points, while MMLU decreases by only
0.11 percentage points. The WMDP reduction is therefore concentrated primarily
on the Bio forget domain.
