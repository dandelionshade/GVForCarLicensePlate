# 车牌识别系统 - 使用指南

## 📋 项目优化总结

### 🎯 图片管理优化

为了让车牌图片管理更加便捷，我们对项目做了以下优化：

#### 1. 标准化的目录结构
```
data/
├── test_images/          # 测试图片库 ⭐ 推荐放置车牌图片的位置
├── uploads/              # 用户上传的图片
└── results/              # 识别结果JSON文件
```

#### 2. Web界面增强
- **双标签页设计**：上传识别 + 图片库管理
- **图片库浏览**：缩略图展示，一键识别
- **文件信息**：显示文件大小、修改时间
- **结果保存**：自动保存识别结果到JSON文件

#### 3. 批量处理工具
- **目录批量识别**：支持整个文件夹批量处理
- **进度显示**：实时显示处理进度
- **结果统计**：成功率、处理时间等统计信息
- **结果保存**：个人结果 + 汇总报告

## 🚀 使用方法

### 方法1：Web界面（推荐）

1. **启动服务**：
   ```bash
   python simple_app.py
   ```

2. **访问界面**：
   - 打开浏览器访问：http://localhost:5000
   - 切换到"图片库"标签页
   - 浏览和识别 `data/test_images/` 中的图片

3. **管理图片**：
   - 将车牌图片放入 `data/test_images/` 目录
   - 点击"刷新"按钮更新图片库
   - 点击"识别"按钮识别特定图片

### 方法2：批量处理工具

1. **基本批量识别**：
   ```bash
   python batch_recognition.py data/test_images
   ```

2. **详细输出 + 保存结果**：
   ```bash
   python batch_recognition.py data/test_images --output data/results --save-individual --verbose
   ```

3. **自定义输入目录**：
   ```bash
   python batch_recognition.py "你的图片目录" --output results
   ```

### 方法3：命令行工具

1. **识别单张图片**：
   ```bash
   python cli_simple.py data/test_images/jing_a12345.jpg
   ```

2. **保存结果 + 显示图像**：
   ```bash
   python cli_simple.py data/test_images/jing_a12345.jpg --output result.json --show
   ```

## 📸 图片管理最佳实践

### 推荐的图片存放策略

1. **测试图片库** (`data/test_images/`)：
   - 存放常用的测试车牌图片
   - 按照省份或类型分类命名
   - 例如：`jing_a12345.jpg`, `hu_b67890.jpg`

2. **项目图片** (自定义目录)：
   - 为不同项目创建专门的图片目录
   - 使用批量处理工具进行识别
   - 例如：`project_images/2025_test/`

3. **上传图片** (`data/uploads/`)：
   - 自动保存用户上传的图片
   - 系统自动按时间戳命名

### 图片格式要求

- **支持格式**：JPG、PNG、BMP、TIFF、GIF
- **推荐尺寸**：建议宽度 > 400px，高度 > 100px
- **图片质量**：清晰度越高，识别效果越好
- **车牌位置**：车牌区域尽量完整，避免遮挡

## 🔧 系统配置

### OCR引擎配置

1. **Tesseract**（默认）：
   - 需要单独安装tesseract程序
   - 支持中文车牌字符识别
   - 配置路径：修改 `simple_app.py` 中的 `tesseract_cmd`

2. **PaddleOCR**（可选）：
   - 基于深度学习，识别精度更高
   - 首次使用会自动下载模型
   - 安装：`pip install paddlepaddle paddleocr`

### 目录自定义

如果需要修改默认目录，编辑 `simple_app.py` 中的路径配置：

```python
# 项目路径配置
PROJECT_ROOT = Path(__file__).parent
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "test_images"        # 测试图片目录
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"                # 上传图片目录
RESULTS_DIR = PROJECT_ROOT / "data" / "results"                # 结果保存目录
```

## 📊 结果分析

### 识别结果格式

识别结果保存为JSON格式，包含以下信息：

```json
{
  "success": true,
  "plate_number": "京A12345",
  "confidence": 0.85,
  "bbox": [50, 50, 300, 100],
  "timestamp": "2025-07-09T10:30:00",
  "processing_time": 0.234,
  "image_name": "test_plate.jpg"
}
```

### 批量处理报告

批量处理会生成汇总报告，包含：

- 处理统计（总数、成功数、失败数）
- 成功率分析
- 处理时间统计
- 详细的每张图片结果

## 🛠️ 故障排除

### 常见问题

1. **OCR引擎不可用**
   - 检查Tesseract是否正确安装
   - 确认tesseract可执行文件路径

2. **识别精度低**
   - 确保图像清晰，车牌区域完整
   - 尝试调整图像亮度和对比度
   - 使用多个OCR引擎进行对比

3. **图片库不显示**
   - 检查 `data/test_images/` 目录是否存在
   - 确认图片格式是否支持
   - 点击"刷新"按钮更新图片库

### 性能优化建议

1. **图片预处理**：
   - 适当调整图片大小（推荐宽度800-1200px）
   - 增强图像对比度
   - 去除噪点

2. **批量处理**：
   - 使用 `--verbose` 参数查看详细处理信息
   - 对于大量图片，建议分批处理
   - 定期清理结果文件避免磁盘空间不足

## 🔄 项目升级

当前版本已经大幅简化了原项目结构，如果需要进一步定制：

1. **添加新的OCR引擎**：在 `SimplePlateRecognizer` 类中添加新方法
2. **自定义预处理**：修改 `preprocess_plate` 方法
3. **扩展Web界面**：在HTML模板中添加新功能
4. **数据库集成**：可以考虑将结果保存到数据库中

这个优化版本既保持了功能的完整性，又大大简化了使用复杂度，特别适合快速部署和测试使用。
