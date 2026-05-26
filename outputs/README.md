# Outputs Directory

这个目录只作为实验输出结构说明和轻量占位。真实权重、预测图、验证集叠加图和指标文件建议放在仓库外部：

```text
D:\SegRuns\outputs\<model_name>\<dataset_name>\<experiment_name>
```

如果临时放在仓库内，建议结构保持一致：

```text
outputs/
└─ deeplabv3plus-grape/
   └─ grape_voc2_iter1/
      └─ exp03_cbam_ppm_focal/
         ├─ weights/
         ├─ metrics/
         └─ val_vis/
```

不要提交真实训练权重、预测图片、mask、CSV 指标表或批量可视化结果。
