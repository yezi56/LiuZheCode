# Outputs Directory

这个目录预留给统一实验输出。

建议存放：

- 权重文件
- 验证集可视化结果
- 指标汇总
- 对比图表

建议结构：

```text
outputs/
└─ model_name/
   └─ dataset_name/
      └─ experiment_name/
         ├─ weights/
         ├─ metrics/
         └─ val_vis/
```
