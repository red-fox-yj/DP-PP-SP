# DP-PP-SP
# 任务描述
GLUE/QQP（Quora Question Pairs）
- 任务类型：二分类
- 数据集来源：Quora 平台的问答对。
- 示例：
```
Question1: "How do I improve my programming skills?"
Question2: "What can I do to become better at programming?"
Label: 1 (重复问题)
```

# 模型架构
BERT Encoder
  ↓
[CLS] Token Embedding
  ↓
Classification Head (Linear Layer)
  ↓
Logits (size = num_labels)

# 实验结果
## 总体结果
| Experiment      | Total Training Time (s) | Throughput (samples/s) | Max GPU Memory (GB) |
|-----------------|--------------------------|-------------------------|----------------------|
| No TP/PP        | 209.71                  | 238.43                 | 3.50                |
| TP Only         | 210.43                  | 237.61                 | 5.13                |
| PP Only         | 211.26                  | 236.67                 | 5.13                |
| TP and PP       | 210.11                  | 237.97                 | 5.13                |

## 详细结果
No TP/PP
| Epoch | Training Loss | Validation Loss | Validation Accuracy | Epoch Time (s) |
|-------|---------------|-----------------|---------------------|----------------|
| 1     | 0.6139        | 0.5376          | 0.69                | 42.26          |
| 2     | 0.3779        | 0.4132          | 0.805               | 41.97          |
| 3     | 0.2235        | 0.4199          | 0.826               | 41.64          |
| 4     | 0.0888        | 0.5952          | 0.822               | 41.97          |
| 5     | 0.0527        | 0.6221          | 0.812               | 41.85          |

TP Only
| Epoch | Training Loss | Validation Loss | Validation Accuracy | Epoch Time (s) |
|-------|---------------|-----------------|---------------------|----------------|
| 1     | 0.0755        | 0.6338          | 0.823               | 42.20          |
| 2     | 0.0125        | 0.7925          | 0.824               | 41.87          |
| 3     | 0.0058        | 1.0041          | 0.81                | 41.96          |
| 4     | 0.0051        | 0.9641          | 0.819               | 42.26          |
| 5     | 0.0151        | 0.8852          | 0.8                 | 42.15          |

PP Only
| Epoch | Training Loss | Validation Loss | Validation Accuracy | Epoch Time (s) |
|-------|---------------|-----------------|---------------------|----------------|
| 1     | 0.0495        | 0.8433          | 0.811               | 42.22          |
| 2     | 0.0025        | 1.0821          | 0.807               | 42.29          |
| 3     | 0.0025        | 1.0177          | 0.818               | 42.50          |
| 4     | 0.0016        | 1.0367          | 0.819               | 42.36          |
| 5     | 0.0112        | 0.8813          | 0.819               | 41.89          |

TP and PP
| Epoch | Training Loss | Validation Loss | Validation Accuracy | Epoch Time (s) |
|-------|---------------|-----------------|---------------------|----------------|
| 1     | 0.0265        | 0.9646          | 0.809               | 42.06          |
| 2     | 0.0008        | 1.1380          | 0.825               | 42.01          |
| 3     | 0.0009        | 1.0919          | 0.825               | 41.93          |
| 4     | 0.0002        | 1.1479          | 0.829               | 42.10          |
| 5     | 0.0002        | 1.1961          | 0.827               | 42.01          |

![training_loss_vs_epochs](https://github.com/user-attachments/assets/8d99bf7f-6876-42ad-9590-9b40f396fc80)

![validation_loss_vs_epochs](https://github.com/user-attachments/assets/c85a4ab5-cacd-4a30-af41-80155dfd422f)

![validation_accuracy_vs_epochs](https://github.com/user-attachments/assets/9836e4f9-cfe1-41a0-a1e8-6acd989d2a3d)

