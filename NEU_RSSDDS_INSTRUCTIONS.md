# NEU-RSSDDS-AUG 缺陷检测训练与测试指南

本文档提供了在NEU-RSSDDS-AUG数据集上使用MM-SAM框架进行RGB-D缺陷检测的完整指南。

## 目录

1. [环境配置](#环境配置)
2. [数据准备](#数据准备)
3. [训练](#训练)
4. [测试](#测试)
5. [文件结构](#文件结构)
6. [故障排除](#故障排除)

## 环境配置

### 系统要求

- Python 3.8+
- CUDA 11.8+ (生产环境) 或 CPU (开发环境)
- 至少 8GB RAM
- 至少 10GB 可用磁盘空间

### 第三方库依赖

#### 核心依赖
```bash
torch==2.1.2
torchvision==0.16.2
numpy>=1.21.0
Pillow>=8.0.0
opencv-python
```

#### 数据处理依赖
```bash
gdal==3.8.3
ruamel.yaml
tqdm>=4.60.0
humanfriendly>=10.0
```

#### 其他依赖
```bash
packaging>=21.0
scikit-learn
matplotlib
seaborn
```

### 自动环境配置

使用提供的自动配置脚本：

```bash
# 1. 创建conda环境
python setup_neu_rssdds.py --env-name mm_sam_neu

# 2. 激活环境
conda activate mm_sam_neu

# 3. 安装依赖和下载模型
python setup_neu_rssdds.py --skip-conda

# 4. 验证安装
python validate_neu_rssdds.py
```

### 手动环境配置

如果自动配置失败，可以手动配置：

```bash
# 1. 创建conda环境
conda create -n mm_sam_neu python=3.10 -y
conda activate mm_sam_neu

# 2. 安装PyTorch (CUDA 11.8)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装项目
pip install -e .

# 4. 安装GDAL
conda install -c conda-forge gdal==3.8.3 -y

# 5. 安装其他依赖
pip install ruamel.yaml Pillow tqdm humanfriendly packaging

# 6. 下载SAM预训练模型
python download_sam_models.py
```

## 数据准备

### 数据集结构

NEU-RSSDDS-AUG数据集应按以下结构组织：

```
../datasets/NEU-RSDDS-AUG/
├── Image_train/          # 训练RGB图像 (.bmp格式)
├── Depth_train/          # 训练深度图像 (.tiff格式)
├── GT_train/             # 训练标注图像 (.png格式)
├── Image_test/           # 测试RGB图像 (.bmp格式)
├── Depth_test/           # 测试深度图像 (.tiff格式)
└── metadata/             # 自动生成的元数据文件
    ├── train.json
    └── test.json
```

### 数据集设置

运行数据集设置脚本：

```bash
python -m pyscripts.neu_rssdds_setup
```

此脚本将：
- 扫描数据集目录
- 创建训练/测试文件映射
- 生成元数据文件
- 如果数据集不存在，创建随机测试数据

### 数据格式要求

- **RGB图像**: .bmp格式，任意尺寸
- **深度图像**: .tiff格式，与对应RGB图像尺寸相同
- **标注图像**: .png格式，二值图像（0=背景，255=缺陷）

## 训练

### 基本训练

使用默认参数进行训练：

```bash
python train_neu_rssdds.py
```

### 自定义训练参数

```bash
python train_neu_rssdds.py \
    --epochs 100 \
    --batch-size 4 \
    --lr 2e-3 \
    --device cuda
```

### 开发模式训练

在开发环境中使用CPU进行训练：

```bash
python train_neu_rssdds.py --dev
```

### 训练参数说明

- `--epochs`: 训练轮数 (默认: 50)
- `--batch-size`: 批次大小 (默认: 2)
- `--lr`: 学习率 (默认: 1.6e-3)
- `--device`: 设备类型 (cuda/cpu)
- `--dev`: 开发模式 (使用./hy-tmp路径和CPU)

### 训练输出

训练过程中会生成以下文件：

- **Checkpoint**: `/hy-tmp/output/checkpoint.pth` (每轮覆盖保存)
- **日志**: `/hy-tmp/output/result.log`
- **实验数据**: `/hy-tmp/experiments/cm_transfer/neu_rssdds_1gpu/`

## 测试

### 基本测试

使用训练好的模型进行测试：

```bash
python test_neu_rssdds.py
```

### 指定checkpoint测试

```bash
python test_neu_rssdds.py --checkpoint /path/to/checkpoint.pth
```

### 开发模式测试

```bash
python test_neu_rssdds.py --dev
```

### 测试参数说明

- `--checkpoint`: checkpoint文件路径 (默认: /hy-tmp/output/checkpoint.pth)
- `--batch-size`: 测试批次大小 (默认: 1)
- `--device`: 设备类型 (cuda/cpu)
- `--dev`: 开发模式

### 测试输出

测试完成后会生成：

- **预测图像**: `/hy-tmp/output/predictions/` (PNG格式，原始文件名)
- **测试日志**: `/hy-tmp/output/result.log`

预测图像格式：
- 二值图像：缺陷=白色(255)，背景=黑色(0)
- 尺寸：恢复到原始图像尺寸

## 文件结构

### 项目结构

```
mm-sam/
├── config/
│   └── cm_transfer/
│       └── neu_rssdds_1gpu.yaml      # 训练配置
├── mm_sam/
│   ├── datasets/
│   │   └── neu_rssdds.py             # 数据集类
│   └── train_agents/
│       └── cm_transfer/
│           └── neu_rssdds.py         # 训练代理
├── pyscripts/
│   └── neu_rssdds_setup.py           # 数据集设置脚本
├── utilbox/
│   ├── global_config.py              # 全局配置
│   └── neu_rssdds_logger.py          # 日志系统
├── pretrained/                       # SAM预训练模型目录
│   └── sam_vit_b_01ec64.pth          # SAM ViT-B模型
├── train_neu_rssdds.py               # 训练脚本
├── test_neu_rssdds.py                # 测试脚本
├── validate_neu_rssdds.py            # 验证脚本
├── setup_neu_rssdds.py               # 环境配置脚本
└── download_sam_models.py            # 模型下载脚本
```

### 输出结构

```
/hy-tmp/output/                       # 生产环境
./hy-tmp/output/                      # 开发环境
├── checkpoint.pth                    # 模型checkpoint
├── result.log                        # 训练/测试日志
└── predictions/                      # 预测结果
    ├── sample_001.png
    ├── sample_002.png
    └── ...
```

## 故障排除

### 常见问题

#### 1. 模块导入错误

```
ModuleNotFoundError: No module named 'ruamel'
```

**解决方案**:
```bash
pip install ruamel.yaml
```

#### 2. SAM模型文件缺失

```
FileNotFoundError: sam_vit_b_01ec64.pth
```

**解决方案**:
```bash
python download_sam_models.py
```

#### 3. CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**:
- 减小批次大小: `--batch-size 1`
- 使用CPU: `--device cpu`
- 使用开发模式: `--dev`

#### 4. 数据集路径错误

```
FileNotFoundError: Metadata file not found
```

**解决方案**:
```bash
python -m pyscripts.neu_rssdds_setup
```

#### 5. 权限错误 (Linux/Mac)

```
PermissionError: [Errno 13] Permission denied: '/hy-tmp'
```

**解决方案**:
```bash
sudo mkdir -p /hy-tmp
sudo chown $USER:$USER /hy-tmp
```

或使用开发模式:
```bash
python train_neu_rssdds.py --dev
```

### 调试技巧

#### 1. 验证安装

```bash
python validate_neu_rssdds.py
```

#### 2. 检查日志

```bash
tail -f /hy-tmp/output/result.log    # 生产环境
tail -f ./hy-tmp/output/result.log   # 开发环境
```

#### 3. 测试单个组件

```python
# 测试数据集加载
from mm_sam.datasets.neu_rssdds import NEURSSDDSDataset
dataset = NEURSSDDSDataset(is_train=True)
print(f"Dataset size: {len(dataset)}")

# 测试模型初始化
from mm_sam.train_agents.cm_transfer.neu_rssdds import NEURSSDDSCMTransferSAM
agent = NEURSSDDSCMTransferSAM(device=torch.device('cpu'))
```

### 性能优化

#### 1. 训练加速

- 使用更大的批次大小 (如果GPU内存允许)
- 启用混合精度训练: `--use_amp True`
- 使用多GPU (修改配置文件中的gpu_num)

#### 2. 内存优化

- 减小批次大小
- 使用梯度累积
- 定期清理GPU缓存

#### 3. 存储优化

- 定期清理临时文件
- 使用SSD存储数据集
- 压缩预测结果

## 高级用法

### 自定义配置

修改 `config/cm_transfer/neu_rssdds_1gpu.yaml`:

```yaml
gpu_num: 1
train_epoch_num: 100          # 增加训练轮数
train_bs: 4                   # 增加批次大小
valid_bs: 8
test_bs: 1

best_model_selection:
  - mean_nonzero_fore_iu
  - max

train_agent: mm_sam.train_agents.cm_transfer.neu_rssdds.NEURSSDDSCMTransferSAM
agent_kwargs:
  train_transforms: cmtransfer_v1
  valid_transforms: resize_1024
  test_transforms: resize_1024
  x_lora_rank: 8              # 增加LoRA rank
```

### 批量处理

```bash
# 批量训练不同配置
for lr in 1e-3 2e-3 5e-3; do
    python train_neu_rssdds.py --lr $lr --epochs 30
    mv /hy-tmp/output/checkpoint.pth /hy-tmp/output/checkpoint_lr${lr}.pth
done
```

### 结果分析

```python
# 分析预测结果
import os
from PIL import Image
import numpy as np

pred_dir = "/hy-tmp/output/predictions"
for filename in os.listdir(pred_dir):
    if filename.endswith('.png'):
        pred = np.array(Image.open(os.path.join(pred_dir, filename)))
        defect_ratio = np.sum(pred > 0) / pred.size
        print(f"{filename}: {defect_ratio:.3f} defect ratio")
```

## 联系与支持

如果遇到问题，请：

1. 首先运行验证脚本: `python validate_neu_rssdds.py`
2. 检查日志文件: `/hy-tmp/output/result.log`
3. 确认数据集格式和路径正确
4. 尝试开发模式进行调试

---

**注意**: 本实现基于MM-SAM框架，专门针对NEU-RSSDDS-AUG数据集进行了优化。在使用其他数据集时可能需要相应的修改。