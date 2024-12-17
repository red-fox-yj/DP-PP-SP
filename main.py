import torch
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from datasets import load_dataset
import deepspeed

# 配置参数
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 5
LR = 2e-5

# 数据加载函数
def load_data(sample_size):
    dataset = load_dataset("glue", "qqp")
    # train_data = dataset["train"].train_test_split(train_size=sample_size, seed=42)["train"]
    # val_data = dataset["validation"].train_test_split(train_size=sample_size // 10, seed=42)["train"]
    train_data = dataset["train"]
    val_data = dataset["validation"]
    print(f"Training set size: {len(train_data)} samples")
    print(f"Validation set size: {len(val_data)} samples")
    return train_data, val_data

# 自定义数据集类
class SentencePairDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1 = self.data[idx]["question1"]
        sentence2 = self.data[idx]["question2"]
        label = self.data[idx]["label"]

        encoding = self.tokenizer(
            sentence1, sentence2,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# DeepSpeed 配置
def get_deepspeed_config(tp_size=None, pp_size=None):
    config = {
        "train_batch_size": BATCH_SIZE,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 4,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 2},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": LR, "betas": [0.9, 0.999], "eps": 1e-8}
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": LR,
                "warmup_num_steps": 100
            }
        }
    }
    if tp_size:
        config["tensor_parallel"] = {"tp_size": tp_size}
    if pp_size:
        config["pipeline"] = {"pipeline_stage": pp_size}
    return config

# 模型评估
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["label"].cuda()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# 模型训练
def train_model(model_engine, dataloader, num_epochs=1, val_dataloader=None):
    model_engine.train()
    total_time = 0
    epoch_results = []

    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()
        # tqdm 进度条
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["label"].cuda()

            # 前向传播
            outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # 更新进度条信息
            progress_bar.set_postfix({"Batch Loss": loss.item()})

            # 反向传播和优化
            model_engine.backward(loss)
            model_engine.step()

        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time
        avg_loss = total_loss / len(dataloader)

        # 验证集评估
        val_loss, val_accuracy = evaluate_model(model_engine, val_dataloader) if val_dataloader else (None, None)

        # 保存结果
        epoch_results.append({
            "Epoch": epoch + 1,
            "Training Loss": avg_loss,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy,
            "Epoch Time (s)": epoch_time
        })
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss}, Val Acc: {val_accuracy}")
    return epoch_results, total_time

# 主函数
def main():
    train_data, val_data = load_data(10000)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 数据集准备
    train_dataset = SentencePairDataset(tokenizer, train_data, MAX_LEN)
    val_dataset = SentencePairDataset(tokenizer, val_data, MAX_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    experiments = [
        {"name": "No TP/PP", "tp_size": None, "pp_size": None},
        {"name": "TP Only", "tp_size": 4, "pp_size": None},
        {"name": "PP Only", "tp_size": None, "pp_size": 2},
        {"name": "TP and PP", "tp_size": 4, "pp_size": 2},
    ]

    results = []
    for exp in experiments:
        print(f"Starting experiment: {exp['name']}")
        
        # 获取 DeepSpeed 配置
        ds_config = get_deepspeed_config(tp_size=exp["tp_size"], pp_size=exp["pp_size"])
        
        # DeepSpeed 初始化
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params=ds_config,
        )
        
        # 开始训练
        epoch_results, training_time = train_model(
            model_engine, train_dataloader, num_epochs=EPOCHS, val_dataloader=val_dataloader
        )
        
        # 计算性能指标
        throughput = (BATCH_SIZE * len(train_dataloader) * EPOCHS) / training_time
        gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转换为 GB
        
        # 保存实验结果
        results.append({
            "Experiment": exp["name"],
            "Total Training Time (s)": training_time,
            "Throughput (samples/s)": throughput,
            "Max GPU Memory (GB)": gpu_memory,
            "Epoch Results": epoch_results,
        })
        
        # 重置 GPU 内存统计
        torch.cuda.reset_peak_memory_stats()

    # 保存结果到 JSON 文件
    with open("detailed_experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Results saved. Generating plots...")
    generate_plots(results)

# 绘图函数
def generate_plots(results):
    if not results:
        print("No results to plot.")
        return

    metrics = ["Training Loss", "Validation Loss", "Validation Accuracy"]
    epochs = range(1, len(results[0]["Epoch Results"]) + 1)

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for res in results:
            if not res["Epoch Results"]:
                continue
            values = [epoch_result[metric] for epoch_result in res["Epoch Results"]]
            plt.plot(epochs, values, marker="o", label=res["Experiment"])
        plt.title(f"{metric} vs. Epochs")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.grid()
        plt.savefig(f"{metric.replace(' ', '_').lower()}_vs_epochs.png")
        plt.close()

if __name__ == "__main__":
    main()