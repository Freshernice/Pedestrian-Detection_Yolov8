"""
PA-100K 属性分类模型评估脚本
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from train_attr_model import get_model, PA100KDataset, compute_map
from config import (
    DATA_DIR, TEST_CSV, ATTR_MODEL_SAVE_PATH,
    BATCH_SIZE, IMG_SIZE, NORM_MEAN, NORM_STD
)

# 【修正】恢复为与 CSV 真实顺序匹配的切片: [G, A(3), D(3), B]
# 同时修正 Age 的类别名称，CSV 中顺序为 Over60, 18-60, Less18 -> 对应 old, adult, minor
ATTR_DEFS = {
    'gender':    (slice(0, 1), ['male', 'female'], False),
    'age':       (slice(1, 4), ['old', 'adult', 'minor'], True),
    'direction': (slice(4, 7), ['front', 'side', 'back'], True),
    'backpack':  (slice(7, 8), ['no', 'yes'], False),
}

# 【修正】同步调整解码函数的索引和类别映射
def decode_attrs(vals):
    return {
        "gender": "female" if vals[0] > 0.5 else "male",
        "age": ["old", "adult", "minor"][np.argmax(vals[1:4])],
        "direction": ["front", "side", "back"][np.argmax(vals[4:7])],
        "backpack": "yes" if vals[7] > 0.5 else "no",
    }

def plot_cm(y_true, y_pred, classes, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d',
                cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set(xlabel='Predicted', ylabel='True', title=title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  混淆矩阵已保存: {save_path}")

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    output_dir = "output/eval"
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    test_dataset = PA100KDataset(TEST_CSV, DATA_DIR, transform, return_id=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)
    print(f"测试集大小: {len(test_dataset)}")

    model = get_model().to(device)
    model.load_state_dict(torch.load(ATTR_MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    thresholds_path = ATTR_MODEL_SAVE_PATH.replace('.pth', '_thresholds.npy')
    thresholds = np.load(thresholds_path) if os.path.exists(thresholds_path) else np.array([0.5, 0.5])
    print(f"✓ 使用动态阈值: Gender={thresholds[0]:.2f}, Backpack={thresholds[1]:.2f}")

    all_probs, all_labels, all_ids = [], [], []
    print("\n开始推理...")
    with torch.no_grad():
        for i, (imgs, labels, ids) in enumerate(test_loader):
            bin_out, multi_out = model(imgs.to(device))
            
            # 【修正】重排概率输出顺序，与 TARGET_COLS 严格一致: [G, A(3), D(3), B]
            probs = torch.cat([
                torch.sigmoid(bin_out[:, 0:1]),
                torch.softmax(multi_out[:, :3], dim=1),
                torch.softmax(multi_out[:, 3:], dim=1),
                torch.sigmoid(bin_out[:, 1:2])
            ], dim=1).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_ids.extend(ids)
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(test_loader)} 批次")

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    print(f"\n推理完成，共 {len(all_probs)} 个样本")

    print("\n" + "=" * 60)
    macro_map = compute_map(all_labels, all_probs)
    print(f"Macro mAP: {macro_map:.4f}")

    print("\n" + "=" * 60)
    metrics_data = []

    for attr_name, (idx_slice, classes, is_multi) in ATTR_DEFS.items():
        y_true_raw = all_labels[:, idx_slice]
        y_pred_raw = all_probs[:, idx_slice]

        if is_multi:
            y_true = np.argmax(y_true_raw, axis=1)
            y_pred = np.argmax(y_pred_raw, axis=1)
            print(f"\n{attr_name.upper()}:")
            print(classification_report(y_true, y_pred, target_names=classes, digits=4))
            cm_path = os.path.join(output_dir, f"{attr_name}_confusion_matrix.png")
            plot_cm(y_true, y_pred, classes,
                    f"{attr_name.capitalize()} Confusion Matrix", cm_path)
        else:
            y_true = y_true_raw.squeeze()
            thresh = thresholds[0] if attr_name == 'gender' else thresholds[1]
            y_pred = (y_pred_raw.squeeze() > thresh).astype(int)

        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(classes))), average=None
        )
        for j, cls in enumerate(classes):
            print(f"  {attr_name}/{cls:8s} | P: {p[j]:.4f} | R: {r[j]:.4f} | F1: {f1[j]:.4f} | N: {sup[j]}")
            metrics_data.append({'attribute': attr_name, 'class': cls,
                                 'precision': p[j], 'recall': r[j],
                                 'f1': f1[j], 'support': sup[j]})

    print("\n" + "=" * 60)
    print("Macro 平均 F1")
    for name in ['gender', 'age', 'direction', 'backpack']:
        vals = [m['f1'] for m in metrics_data if m['attribute'] == name]
        print(f"  {name:12s}: {np.mean(vals):.4f}")

    metrics_path = os.path.join(output_dir, "metrics.csv")
    pd.DataFrame(metrics_data).to_csv(metrics_path, index=False)
    print(f"\n指标已保存: {metrics_path}")

    map_path = os.path.join(output_dir, "map.csv")
    pd.DataFrame([{'macro_mAP': macro_map}]).to_csv(map_path, index=False)
    print(f"mAP 已保存: {map_path}")

    jsonl_path = os.path.join(output_dir, "predictions.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i in range(len(all_probs)):
            f.write(json.dumps({
                'image_id': all_ids[i],
                'ground_truth': decode_attrs(all_labels[i]),
                'prediction': decode_attrs(all_probs[i]),
                'probabilities': all_probs[i].tolist()
            }, ensure_ascii=False) + '\n')
    print(f"预测记录已保存: {jsonl_path}")
    print("\n评估完成！输出目录:", output_dir)

if __name__ == "__main__":
    evaluate()