from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import torch
import numpy as np
import uuid
import os
import subprocess
import threading
from ultralytics import YOLO
from torchvision import transforms
from train_attr_model import get_model
from config import (
    YOLO_MODEL_PATH, ATTR_MODEL_SAVE_PATH, CONF_THRESHOLD,
    YOLO_CLASSES, CLASS_NAMES, IMG_SIZE, NORM_MEAN, NORM_STD, SKIP_INTERVAL
)

app = FastAPI()

# ================= 模型初始化 =================
print("加载模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLO(YOLO_MODEL_PATH)
attr_model = get_model().to(device)
attr_model.load_state_dict(torch.load(ATTR_MODEL_SAVE_PATH, map_location=device))
attr_model.eval()

attr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

# ================= 属性解析 =================
def parse_attributes(img_crop):
    img_tensor = attr_transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        bin_out, multi_out = attr_model(img_tensor)
        probs = torch.cat([
            torch.sigmoid(bin_out[:, 0:1]),
            torch.softmax(multi_out[:, :3], dim=1),
            torch.softmax(multi_out[:, 3:], dim=1),
            torch.sigmoid(bin_out[:, 1:2])
        ], dim=1).squeeze().cpu().numpy()

    gender = "female" if probs[0] > 0.5 else "male"
    age_probs = probs[1:4]
    age_idx = np.argmax(age_probs)
    age = ["old", "adult", "minor"][age_idx]
    dir_probs = probs[4:7]
    dir_idx = np.argmax(dir_probs)
    direction = ["front", "side", "back"][dir_idx]
    backpack = "yes" if probs[7] > 0.5 else "no"

    return {"gender": gender, "age": age, "direction": direction, "backpack": backpack}

# ================= 文件存储 =================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ================= 处理状态存储 =================
video_status = {}

# ================= 异步转码函数 =================
def transcode_video(video_id, raw_path, video_path):
    """后台线程执行转码"""
    try:
        # 使用 ultrafast preset 加速转码
        result = subprocess.run([
            "ffmpeg", "-i", raw_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-y", video_path
        ], capture_output=True, text=True, timeout=300)

        # 如果带音频失败，尝试无音频
        if result.returncode != 0:
            print(f"带音频转码失败，尝试无音频: {result.stderr[:500]}")
            result = subprocess.run([
                "ffmpeg", "-i", raw_path,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-an",
                "-movflags", "+faststart",
                "-y", video_path
            ], capture_output=True, text=True, timeout=300)

        # 删除原始文件
        if os.path.exists(raw_path):
            os.remove(raw_path)

        if result.returncode != 0:
            video_status[video_id] = {"status": "error", "message": "转码失败"}
        else:
            video_status[video_id] = {"status": "ready"}

    except subprocess.TimeoutExpired:
        video_status[video_id] = {"status": "error", "message": "转码超时"}
    except Exception as e:
        video_status[video_id] = {"status": "error", "message": str(e)}

# ================= 路由 =================
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    raw_path = os.path.join(UPLOAD_DIR, f"{video_id}_raw")
    video_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")

    # 保存原始文件
    with open(raw_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 初始化状态：转码中
    video_status[video_id] = {"status": "transcoding"}

    # 启动后台线程转码
    thread = threading.Thread(target=transcode_video, args=(video_id, raw_path, video_path))
    thread.start()

    return {"video_id": video_id}

@app.get("/status/{video_id}")
async def get_status(video_id: str):
    return video_status.get(video_id, {"status": "not_found"})

@app.websocket("/ws/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    await websocket.accept()
    video_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")

    # 等待文件就绪
    for _ in range(50):  # 最多等待 5 秒
        if os.path.exists(video_path):
            break
        import time
        time.sleep(0.1)

    if not os.path.exists(video_path):
        await websocket.close()
        return

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    last_frame_targets = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            is_skipped = (frame_id % SKIP_INTERVAL != 0)
            current_targets = []

            if not is_skipped:
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

                    if cls_name == "person":
                        h, w = frame.shape[:2]
                        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                        if cx2 > cx1 and cy2 > cy1:
                            crop = frame[cy1:cy2, cx1:cx2]
                            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            target_data["attributes"] = parse_attributes(crop)
                        else:
                            target_data["attributes"] = {"gender": "unknown", "age": "unknown", "direction": "unknown", "backpack": "unknown"}

                    current_targets.append(target_data)

                last_frame_targets = current_targets
            else:
                current_targets = last_frame_targets

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            data = {
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "targets": current_targets
            }

            await websocket.send_json(data)
            frame_id += 1

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        cap.release()
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)