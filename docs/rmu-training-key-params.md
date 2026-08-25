# RMU / Training Key Parameters

## RMU

| 项 | 值 |
| --- | --- |
| 计算 RMU loss 的层 | 第 10 层 |
| 实际训练的层 | 第 8、9、10 层 |
| 实际训练的模块 | MoE 的 expert / shared expert MLP 参数 |
| 遗忘损失权重 | 350 |
| 保留损失权重 | 1 |
| 控制向量强度 | 300 |

## 训练

| 项 | 值 |
| --- | --- |
| GPU 数 | 5 张 |
| 单卡 batch size | 8 |
| 梯度累计 | 2 |
| 有效 batch size | 80 |
| epoch | 1 |
| 学习率 | 1e-5 |
| 精度 | bf16 |
| warmup | 0 |
| 梯度检查点 | 关闭 |

## 数据

| 项 | 值 |
| --- | --- |
| 遗忘数据 | WMDP-Bio forget |
| 保留数据 | FineWeb-Edu retain / WMDP retain |
| 最大长度 | 1024 |
| chunk stride | 1024 |
| 最小 chunk 长度 | 1 |
| 是否丢弃最后不足长 chunk | 否 |
