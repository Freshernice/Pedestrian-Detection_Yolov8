import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from train_attr_model import get_model # 复用训练脚本的配置

# ================= 配置区 =================

from config import (
    VIDEO_PATH, VIDEO_SOURCE_NAME, JSONL_OUTPUT_PATH, YOLO_MODEL_PATH, ATTR_MODEL_SAVE_PATH,
    SKIP_INTERVAL, CONF_THRESHOLD, YOLO_CLASSES, CLASS_NAMES, IMG_SIZE,NORM_MEAN, NORM_STD
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 初始化模型 =================
print("加载 YOLO 模型...")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("加载属性分类模型...")
attr_model = get_model().to(device)
attr_model.load_state_dict(torch.load(ATTR_MODEL_SAVE_PATH, map_location=device))
attr_model.eval()

attr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

# ================= 属性解析逻辑 =================
def parse_attributes(img_crop):
    """将裁剪的人物图像输入模型，解析为 JSON 格式的属性"""
    img_tensor = attr_transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = attr_model(img_tensor).squeeze().cpu().numpy()
    
    probs = 1 / (1 + np.exp(-logits)) # Sigmoid
    
    # 映射规则
    gender = "female" if probs[0] > 0.5 else "male"
    
    age_probs = probs[1:4] # Less18, 18-60, Over60
    age_idx = np.argmax(age_probs)
    age = ["minor", "adult", "old"][age_idx]
    
    dir_probs = probs[4:7] # Front, Side, Back
    dir_idx = np.argmax(dir_probs)
    direction = ["front", "side", "back"][dir_idx]
    
    backpack = "yes" if probs[7] > 0.5 else "no"
    
    return {
        "gender": gender,
        "age": age,
        "direction": direction,
        "backpack": backpack
    }

# ================= 视频处理主循环 =================
cap = cv2.VideoCapture(VIDEO_PATH)
global_frame_id = 0
last_frame_targets = [] # 用于跳帧复用的缓存

print(f"开始处理视频: {VIDEO_PATH} -> {JSONL_OUTPUT_PATH}")
with open(JSONL_OUTPUT_PATH, 'w', encoding='utf-8') as f_jsonl:
    while cap.isOpened():
        ret, frame = cap.read() # 是否成功读取到一帧。True = 成功，False = 视频结束或读取失败
        if not ret:
            break

        is_skipped = (global_frame_id % SKIP_INTERVAL != 0) # 跳帧逻辑 帧号对跳帧间隔取模，如果余数不为 0，说明是跳帧
        
        current_targets = []

        if not is_skipped:
            # --- 正常推理帧 ---
            # 使用 track 保持 ID 连续性（虽然这里我们直接复用结果，但 track 有助于后续扩展）
            results = yolo_model.track(frame, persist=True, classes=YOLO_CLASSES, conf=CONF_THRESHOLD, verbose=False)
            
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASS_NAMES[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                target_data = {
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "attributes": {}
                }
                
                # 只有 person 才进行属性识别
                if cls_name == "person":
                    # 边界保护
                    h, w = frame.shape[:2]
                    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    if cx2 > cx1 and cy2 > cy1:
                        crop = frame[cy1:cy2, cx1:cx2]
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # BGR → RGB
                        target_data["attributes"] = parse_attributes(crop)
                    else:
                        target_data["attributes"] = {"gender": "unknown", "age": "unknown", "direction": "unknown", "backpack": "unknown"}
                
                current_targets.append(target_data)
            
            # 更新缓存，供跳帧使用
            last_frame_targets = current_targets 
        else:
            # --- 跳帧复用 ---
            current_targets = last_frame_targets

        # 构建 JSONL 行数据
        jsonl_record = {
            "source": VIDEO_SOURCE_NAME,
            "frame_id": global_frame_id,
            "timestamp_ms": int(cap.get(cv2.CAP_PROP_POS_MSEC)),
            "is_skipped": is_skipped,
            "targets": current_targets
        }
        
        # 写入文件
        f_jsonl.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')
        global_frame_id += 1

cap.release()
print(f"处理完成！共生成 {global_frame_id} 帧数据，保存至 {JSONL_OUTPUT_PATH}")