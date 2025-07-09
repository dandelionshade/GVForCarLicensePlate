# 车牌识别系统 - 快速开始

## 项目重组完成 ✅

重组时间: 2025-07-09 15:47:25

## 📁 标准目录结构

```
GVForCarLicensePlate/
├── simple_app.py              # 🌐 Web应用主文件
├── cli_simple.py             # 🖥️ 命令行工具
├── batch_recognition.py      # 📦 批量处理工具
├── generate_test_images.py   # 🖼️ 测试图片生成
├── requirements_minimal.txt  # 📋 依赖包列表
├── README_minimal.md         # 📖 项目说明
├── USAGE_GUIDE.md           # 📚 详细使用指南
└── data/                    # 📂 数据目录
    ├── test_images/         # 🖼️ 测试图片库（推荐放置车牌图片）
    ├── uploads/             # 📤 上传图片存储
    └── results/             # 📊 识别结果保存
```

## 🚀 快速启动

1. **安装依赖**：
   ```bash
   pip install -r requirements_minimal.txt
   ```

2. **生成测试图片**：
   ```bash
   python generate_test_images.py
   ```

3. **启动Web服务**：
   ```bash
   python simple_app.py
   ```
   访问: http://localhost:5000

4. **命令行识别**：
   ```bash
   python cli_simple.py data/test_images/jing_a12345.jpg
   ```

5. **批量处理**：
   ```bash
   python batch_recognition.py data/test_images --verbose
   ```

## 🎯 车牌图片管理

### 推荐做法：
- 将车牌图片放入 `data/test_images/` 目录
- 使用Web界面的"图片库"功能浏览和识别
- 支持格式：JPG、PNG、BMP、TIFF等

### 批量处理：
- 使用 `batch_recognition.py` 处理整个文件夹
- 自动生成识别报告和统计信息
- 支持保存个人结果文件

## 📊 项目优化成果

- ✅ 文件数量：从 17695 个文件精简为 9 个核心文件
- ✅ 目录结构：标准化数据目录管理
- ✅ 功能完整：保留所有核心识别功能
- ✅ 易于使用：Web界面 + 命令行 + 批量处理
- ✅ 结果管理：自动保存识别结果和统计信息

## 🔧 下一步

1. 查看详细使用指南：`USAGE_GUIDE.md`
2. 测试系统功能：`python test_simple.py`
3. 根据需要放置车牌图片到 `data/test_images/`
4. 开始使用车牌识别功能！

---
💡 提示：如果需要删除旧的复杂文件，可以运行 `clean_project.py`
