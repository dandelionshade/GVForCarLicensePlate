# 车牌识别系统 - 简化版本

最小可行的车牌识别系统，包含Web界面和命令行工具。

## 核心功能

- 车牌区域检测
- 多OCR引擎识别（Tesseract、PaddleOCR）
- Web界面上传识别
- 命令行批处理
- API接口

## 文件结构

```
├── simple_app.py           # 主应用文件（Web + API）
├── cli_simple.py          # 命令行工具
├── requirements_minimal.txt # 最小依赖包
└── README_minimal.md      # 说明文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_minimal.txt
```

### 2. 启动Web服务

```bash
python simple_app.py
```

访问 http://localhost:5000

### 3. 命令行使用

```bash
# 识别单张图片
python cli_simple.py image.jpg

# 保存结果到JSON
python cli_simple.py image.jpg --output result.json

# 显示检测结果
python cli_simple.py image.jpg --show
```

## API接口

### 车牌识别
- **URL**: `/api/recognize`
- **方法**: POST
- **参数**: `image` (文件上传)
- **返回**: JSON格式识别结果

### 健康检查
- **URL**: `/api/health`
- **方法**: GET
- **返回**: 系统状态信息

## 支持的OCR引擎

1. **Tesseract** - 开源OCR引擎
   - 需要安装tesseract程序
   - 支持中文车牌字符

2. **PaddleOCR** - 百度开源OCR
   - 基于深度学习
   - 识别精度更高

## 配置说明

在`simple_app.py`中可以调整：

- Tesseract路径: `pytesseract.pytesseract.tesseract_cmd`
- 车牌检测参数: 宽高比、面积阈值等
- OCR识别参数: 字符白名单、识别模式等

## 注意事项

1. 确保已正确安装Tesseract程序
2. 首次使用PaddleOCR会自动下载模型
3. 图像质量影响识别准确度
4. 支持常见图像格式：JPG、PNG、BMP等

## 系统要求

- Python 3.7+
- OpenCV 4.0+
- 至少4GB内存（使用PaddleOCR时）

## 故障排除

### OCR引擎不可用
- 检查Tesseract是否正确安装
- 检查PaddleOCR依赖是否完整

### 识别精度低
- 确保图像清晰，车牌区域完整
- 尝试调整图像亮度和对比度
- 使用多个OCR引擎进行对比
