# 视觉检测项目

## 1. 项目概述

本项目是一个基于 Python 的视觉检测与行人属性识别项目，主要包含两部分能力：

- **在线视频检测服务**：使用 FastAPI 提供 HTTP 接口和 WebSocket 推送能力，支持上传视频、后台转码、逐帧读取视频，并通过 YOLO 模型检测 `person / bicycle / car` 目标。对于检测到的行人目标，进一步调用 ResNet18 多任务属性模型，识别性别、年龄段、朝向和是否背包。
- **离线训练与评估流程**：基于 PA-100K 行人属性数据集训练多任务属性分类模型，并生成模型权重、阈值文件和评估结果，包括 mAP、Precision / Recall / F1、混淆矩阵和逐样本预测记录。

当前代码覆盖 `app.py`、`train_attr_model.py`、`eval.py` 和 `config.py` 中的逻辑。前端页面 `index.html` 由 `app.py` 读取，但当前代码中未包含具体实现。

---

## 2. 核心功能

| 功能模块 | 说明 |
|---|---|
| 视频上传 | `POST /upload` 接收上传视频，生成唯一 `video_id`，并将原始文件保存到 `uploads/` |
| 视频转码 | 使用 `ffmpeg` 将上传视频后台转码为浏览器友好的 `.mp4` 格式 |
| 转码状态查询 | `GET /status/{video_id}` 查询当前视频处理状态 |
| WebSocket 实时推送 | `WS /ws/{video_id}` 逐帧读取视频并推送检测结果 |
| 目标检测与跟踪 | 使用 Ultralytics YOLO 进行 `track`，检测类别来自 `config.py` 中的 [YOLO_CLASSES]
| 行人属性识别 | 对 `person` 目标裁剪图像，输入 ResNet18 多任务属性模型，输出性别、年龄、朝向、背包 |
| 属性模型训练 | 使用 PA-100K 数据集训练 `gender / age / direction / backpack` 四类属性 |
| 模型评估 | 计算 Macro mAP、逐类 P/R/F1、生成混淆矩阵和预测记录文件 |

---

## 3. 技术栈

### Python 依赖

| 库 | 用途 |
|---|---|
| FastAPI | Web API 与 WebSocket 服务 |
| Uvicorn | ASGI 服务器 |
| python-multipart | FastAPI 文件上传解析支持 |
| OpenCV (`cv2`) | 视频读取、图像裁剪、颜色空间转换 |
| PyTorch | 属性模型训练、推理、张量处理 |
| Torchvision | ResNet18 backbone、图像变换 |
| Ultralytics YOLO | 目标检测与跟踪 |
| Pillow / `PIL` | 图像读取与转换 |
| NumPy | 数值计算与数组处理 |
| pandas | 读取 CSV 数据集标注 |
| scikit-learn | mAP、F1、Precision / Recall、混淆矩阵、class weight 计算 |
| Matplotlib / Seaborn | 生成混淆矩阵图 |
| `subprocess`, `threading`, `uuid`, `os` | 文件存储、后台转码、任务状态管理 |

### 外部依赖

| 依赖 | 用途 |
|---|---|
| FFmpeg | 视频转码，由 `app.py` 中的 `subprocess.run` 调用 |

4. 目录结构
视觉检测项目/
├── app.py                         # FastAPI 服务入口，负责上传、转码、WebSocket 推理
├── train_attr_model.py            # PA-100K 属性模型训练脚本
├── eval.py                        # PA-100K 属性模型评估脚本
├── config.py                      # 路径、训练参数、推理参数、标签配置
│
├── index.html                     # 前端页面，app.py 会读取该文件
├── uploads/                       # 运行服务后自动创建，存放上传视频、转码结果和静态访问目录
│
├── yolov8m.pt                     # YOLOv8 模型权重，由 config.YOLO_MODEL_PATH 指定
├── attr_model.pth                 # 属性分类模型权重，由 config.ATTR_MODEL_SAVE_PATH 指定
├── attr_model_thresholds.npy      # 训练阶段保存的二值属性阈值，可选
│
├── PA-100K/                       # PA-100K 数据集目录
│   ├── data/                      # 图像目录
│   ├── train.csv                  # 训练集标注
│   ├── val.csv                    # 验证集标注
│   └── test.csv                   # 测试集标注
│
└── output/
    └── eval/                      # 评估脚本输出目录
        ├── metrics.csv
        ├── map.csv
        ├── predictions.jsonl
        ├── age_confusion_matrix.png
        └── direction_confusion_matrix.png

5. 系统架构
   
浏览器 / 客户端
    │
    │ POST /upload
    ▼
FastAPI app.py
    │
    ├── 保存原始上传文件到 uploads/{video_id}_raw
    ├── 初始化 video_status[video_id] = {"status": "transcoding"}
    ├── 启动后台线程 transcode_video
    │        │
    │        ▼
    │     FFmpeg
    │        │
    │        ▼
    │   uploads/{video_id}.mp4
    │        │
    │        ▼
    │   更新 video_status[video_id]
    │
    │ GET /status/{video_id}
    ▼
查询转码状态
    │
    │ WebSocket /ws/{video_id}
    ▼
FastAPI WebSocket Endpoint
    │
    ├── 使用 OpenCV 读取 uploads/{video_id}.mp4
    ├── 每隔 SKIP_INTERVAL 帧调用 YOLO.track
    ├── 对 person 目标裁剪图像
    ├── 调用 ResNet18 属性模型
    └── 发送 JSON 检测结果
    │
    ▼
浏览器 / 客户端

5.2 离线训练与评估架构

PA-100K 数据集
train.csv / val.csv / test.csv
        │
        ▼
train_attr_model.py
        │
        ├── 加载数据
        ├── 构建 MultiHeadResNet
        ├── 训练属性模型
        ├── 验证 mAP 与阈值
        └── 保存 attr_model.pth + attr_model_thresholds.npy
                │
                ▼
eval.py
        │
        ├── 加载测试集
        ├── 推理概率
        ├── 计算 mAP
        ├── 计算逐类指标
        ├── 生成混淆矩阵
        └── 输出 CSV / JSONL / PNG

6. 模块说明
   
config.py 集中定义数据集路径、模型路径、训练超参数和推理参数。

路径配置
配置项	说明
PA100K_DIR	PA-100K 数据集根目录，当前值为 "PA-100K"
DATA_DIR	图像目录，当前为 PA-100K/data
TRAIN_CSV	训练集 CSV，当前为 PA-100K/train.csv
VAL_CSV	验证集 CSV，当前为 PA-100K/val.csv
TEST_CSV	测试集 CSV，当前为 PA-100K/test.csv
ATTR_MODEL_SAVE_PATH	属性模型保存路径，当前为 attr_model.pth
当前代码中还存在以下配置：

配置项	说明
JSONL_OUTPUT_PATH	保留字段，当前服务代码中未直接使用
VIDEO_SOURCE_NAME	保留字段，当前服务代码中未直接使用
VIDEO_PATH	保留字段，当前 Web 服务通过上传视频生成路径，未直接使用训练配置

配置项	说明
BATCH_SIZE	训练和评估 batch size
EPOCHS	最大训练轮数
LEARNING_RATE	初始学习率
IMG_SIZE	属性模型输入图像尺寸，当前为 224
NORM_MEAN	图像归一化均值
NORM_STD	图像归一化标准差
TARGET_COLS	PA-100K CSV 中属性列顺序
TARGET_COLS 顺序为：
   ['Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back', 'Backpack']

对应索引如下：

索引	CSV 列	语义
0	Female	性别，1 表示 female，0 表示 male
1	AgeOver60	年龄段：old
2	Age18-60	年龄段：adult
3	AgeLess18	年龄段：minor
4	Front	朝向：front
5	Side	朝向：side
6	Back	朝向：back
7	Backpack	是否背包，1 表示 yes，0 表示 no

推理配置
配置项	说明
YOLO_MODEL_PATH	YOLO 模型权重路径，当前为 yolov8m.pt
CONF_THRESHOLD	YOLO 检测置信度阈值，当前为 0.5
SKIP_INTERVAL	WebSocket 推理间隔，当前为每 3 帧推理一次，其余帧复用上一次结果
YOLO_CLASSES	YOLO 检测类别，当前为 [0, 1, 2]，对应 COCO 中的 person / bicycle / car
CLASS_NAMES	类别 ID 到名称映射

# 6.2 train_attr_model.py
该脚本负责训练行人属性分类模型。

数据集定义
PA100KDataset 读取 PA-100K CSV 文件，每行包含：

Image：图像文件名或相对路径
TARGET_COLS：8 维属性标签
训练和验证图像会按 CSV 中的 Image 与 DATA_DIR 拼接路径读取。

# 模型结构
属性模型为 MultiHeadResNet。

输入图像 224 × 224 × 3
        │
        ▼
ResNet18 backbone
        │
        ▼
512 维特征
        │
        ├───────────────┐
        ▼               ▼
fc_binary 512→2    fc_multi 512→6
        │               │
        ▼               ▼
gender / backpack   age / direction

其中：

fc_binary 输出 2 维，对应 gender 和 backpack
fc_multi 输出 6 维，前 3 维对应 age，后 3 维对应 direction

# 属性建模方式
属性	建模方式	输出维度	预测类别
gender	二分类，sigmoid	1	male / female
age	多分类，softmax	3	old / adult / minor
direction	多分类，softmax	3	front / side / back
backpack	二分类，sigmoid	1	no / yes

损失函数
任务	损失函数	说明
gender	AsymmetricLoss	gamma_neg=4, gamma_pos=0, m=0.05
backpack	AsymmetricLoss	gamma_neg=2, gamma_pos=0, m=0.05
age	CrossEntropyLoss	使用 compute_class_weight('balanced') 计算类别权重
direction	CrossEntropyLoss	使用 compute_class_weight('balanced') 计算类别权重

# 数据增强
训练阶段使用较温和的增强：

变换	说明
Resize	缩放到 IMG_SIZE × IMG_SIZE
RandomHorizontalFlip	水平翻转，概率 0.5
ColorJitter	轻度色彩抖动
ToTensor	转张量
Normalize	使用 ImageNet 均值方差归一化

# 优化器与学习率调度
项目	配置
优化器	AdamW
初始学习率	LEARNING_RATE
weight decay	1e-4
warmup	LinearLR，前 2 个 epoch
主衰减	CosineAnnealingLR
调度组合	SequentialLR
最小学习率	LEARNING_RATE * 0.01

# 验证与早停
验证指标：Macro mAP
早停 patience：7
若验证 mAP 提升，则保存：
attr_model.pth
attr_model_thresholds.npy
保存的阈值当前为 gender 和 backpack 两个二值属性阈值

# 6.3 eval.py
该脚本负责评估训练好的属性模型。

评估输入
评估脚本需要：

TEST_CSV
DATA_DIR
ATTR_MODEL_SAVE_PATH
可选的 attr_model_thresholds.npy
评估指标
指标	说明
Macro mAP	对所有 8 个属性列分别计算 AP 后取平均
Precision / Recall / F1 / Support	对每个属性下的每个类别分别统计
混淆矩阵	对 age 和 direction 这类多分类属性生成 PNG
预测记录	保存每张图像的预测概率、真实标签、预测结果
阈值使用逻辑
评估脚本会读取训练阶段保存的 attr_model_thresholds.npy。

属性	阈值使用方式
gender	使用读取到的阈值进行二值化，计算 P/R/F1
backpack	使用读取到的阈值进行二值化，计算 P/R/F1
age	使用 softmax 概率 argmax
direction	使用 softmax 概率 argmax
需要注意的是，eval.py 中保存逐样本预测结果时使用的 decode_attrs 仍按固定阈值 0.5 解码 gender 和 backpack，该逻辑与动态阈值计算指标存在区分。

输出文件
文件	内容
output/eval/metrics.csv	每个属性、每个类别的 P/R/F1/support
output/eval/map.csv	Macro mAP
output/eval/predictions.jsonl	每行一个样本的 ID、真实属性、预测属性、概率
output/eval/age_confusion_matrix.png	age 混淆矩阵
output/eval/direction_confusion_matrix.png	direction 混淆矩阵

predictions.jsonl 每行大致结构：
{
  "image_id": "xxx.jpg",
  "ground_truth": {
    "gender": "female",
    "age": "adult",
    "direction": "front",
    "backpack": "yes"
  },
  "prediction": {
    "gender": "female",
    "age": "adult",
    "direction": "front",
    "backpack": "no"
  },
  "probabilities": [0.9, 0.01, 0.97, 0.02, 0.95, 0.03, 0.02, 0.4]
}

# 6.4 app.py
app.py 是项目的在线服务入口，基于 FastAPI 实现。

启动逻辑
服务启动时会加载：

YOLO 模型：YOLO_MODEL_PATH
属性模型：ATTR_MODEL_SAVE_PATH
属性图像变换：ToPILImage -> Resize -> ToTensor -> Normalize
推理设备优先使用 CUDA，如果没有可用 GPU 则退回 CPU。

7. 在线服务流程
7.1 视频上传流程

处理过程：

生成 video_id
将文件写入 uploads/{video_id}_raw
初始化状态为 transcoding
启动后台线程调用 FFmpeg 转码
返回 video_id

7.2 视频转码流程
后台转码函数 transcode_video 使用 FFmpeg 将原始视频转码为mp4

状态可能包括：

状态	说明
transcoding	正在转码
ready	转码完成，目标 mp4 可用
error	转码失败或超时
not_found	未找到对应 video_id 的状态记录

7.3 WebSocket 推理流程

处理过程：

接受 WebSocket 连接
等待 uploads/{video_id}.mp4 出现，最多等待 5 秒
如果视频不存在，关闭连接
使用 OpenCV 读取视频帧
每隔 SKIP_INTERVAL 帧执行一次 YOLO 检测与跟踪
其他帧复用上一次目标结果
对 person 目标裁剪人体区域
将裁剪图像输入属性模型
向前端发送该帧结果

8. WebSocket 推送数据格式

{
  "frame_id": 12,
  "timestamp_ms": 500,
  "targets": [
    {
      "class": "person",
      "confidence": 0.72,
      "bbox": [200, 80, 320, 400],
      "attributes": {
        "gender": "female",
        "age": "adult",
        "direction": "front",
        "backpack": "yes"
      }
    },
    {
      "class": "car",
      "confidence": 0.88,
      "bbox": [400, 200, 800, 500],
      "attributes": {}
    }
  ]
}

字段说明：

字段	说明
frame_id	当前视频帧序号
timestamp_ms	当前视频帧时间戳，单位为毫秒
targets	当前帧检测结果列表
targets[].class	检测类别名称
targets[].confidence	检测置信度
targets[].bbox	边界框坐标，顺序为 [x1, y1, x2, y2]
targets[].attributes	属性结果。person 包含属性，其他类别当前为空对象

9. 检测与属性结果说明
9.1 YOLO 检测类别
COCO class id	名称
0	person
1	bicycle
2	car

9.2 行人属性解码

app.py 中 parse_attributes 的解码规则：

属性	判断规则	输出
gender	prob[0] > 0.5 为 female，否则 male	female / male
age	argmax(prob[1:4])	old / adult / minor
direction	argmax(prob[4:7])	front / side / back
backpack	prob[7] > 0.5 为 yes，否则 no	yes / no


10. HTTP / WebSocket 接口汇总
方法	路径	说明
GET	/	返回 index.html
POST	/upload	上传视频，返回 video_id
GET	/status/{video_id}	查询视频转码状态
WebSocket	/ws/{video_id}	连接后逐帧推送检测结果
Static	/uploads/...	静态访问 uploads/ 目录，例如转码后的 mp4

11. 项目文件与职责速查
文件	主要职责
app.py	启动 FastAPI 服务，处理上传、转码状态、WebSocket 推理
config.py	统一维护数据集路径、模型路径、训练参数、推理参数
train_attr_model.py	构建 PA-100K 数据集、训练 ResNet18 多任务属性模型、保存最佳模型和阈值
eval.py	加载测试集，计算指标，生成 CSV、JSONL、混淆矩阵
index.html	前端页面，负责上传视频、接收 WebSocket 数据、展示检测结果
