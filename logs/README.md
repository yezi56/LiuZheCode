# Logs Directory

这个目录只作为日志结构说明和轻量占位。真实训练日志可以放在仓库外部：

```text
D:\SegRuns\logs\<model_name>\<dataset_name>\<experiment_name>
```

如果临时放在仓库内，建议结构保持一致：

```text
logs/
└─ deeplabv3plus-grape/
   └─ grape_voc2_iter1/
      └─ exp03_cbam_ppm_focal/
```

不要提交大批量训练日志、TensorBoard 文件或中间结果。
