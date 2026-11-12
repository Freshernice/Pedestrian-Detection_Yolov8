import os

# ================= 路径配置 =================
# PA-100K 数据集路径
PA100K_DIR = "PA-100K"
DATA_DIR = os.path.join(PA100K_DIR, "data")
TRAIN_CSV = os.path.join(PA100K_DIR, "train.csv")
VAL_CSV = os.path.join(PA100K_DIR, "val.csv")
TEST_CSV = os.path.join(PA100K_DIR, "test.csv")

# 模型与输出路径
ATTR_MODEL_SAVE_PATH = "attr_model.pth"
JSONL_OUTPUT_PATH = "output.jsonl"
VIDEO_SOURCE_NAME = "test_video.mp4" # 对应 JSONL 中的 source 字段

# ================= 训练配置 =================
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0004
IMG_SIZE = 224

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# 属性列定义 (共8维输出)
# CSV顺序: Female, AgeOver60, Age18-60, AgeLess18, Front, Side, Back, Backpack
# 对应索引: 0:Female, 1:AgeOver60(old), 2:Age18-60(adult), 3:AgeLess18(minor), 4:Front, 5:Side, 6:Back, 7:Backpack
TARGET_COLS = ['Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back', 'Backpack']

# ================= 推理配置 =================
VIDEO_PATH = "/root/code/视觉检测项目/行人检测视频01.flv" # 视频路径
YOLO_MODEL_PATH = "yolov8m.pt"
CONF_THRESHOLD = 0.5
SKIP_INTERVAL = 3  # 每 3 帧推理 1 次，其余帧复用

# COCO 类别 ID 映射 (0:person, 1:bicycle, 2:car)
YOLO_CLASSES = [0, 1, 2] 
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car"}