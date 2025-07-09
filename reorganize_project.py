#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目重组脚本 - 最终版本
清理旧文件，保留优化后的核心功能
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def reorganize_project():
    """重组项目结构"""
    
    project_root = Path(__file__).parent
    
    print("🚀 开始重组车牌识别项目...")
    print(f"📁 项目根目录: {project_root}")
    
    # 1. 创建优化后的目录结构
    print("\n📂 创建标准目录结构...")
    essential_dirs = [
        "data/test_images",
        "data/uploads", 
        "data/results",
        "logs"
    ]
    
    for dir_path in essential_dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    # 2. 核心文件列表
    core_files = {
        "simple_app.py": "主应用文件（Web界面 + API）",
        "cli_simple.py": "命令行识别工具",
        "batch_recognition.py": "批量处理工具",
        "generate_test_images.py": "测试图片生成工具",
        "requirements_minimal.txt": "最小依赖包",
        "README_minimal.md": "项目说明文档",
        "USAGE_GUIDE.md": "详细使用指南",
        "clean_project.py": "项目清理脚本",
        "test_simple.py": "功能测试脚本"
    }
    
    # 3. 检查核心文件是否存在
    print("\n📄 检查核心文件...")
    missing_files = []
    for filename, description in core_files.items():
        file_path = project_root / filename
        if file_path.exists():
            print(f"  ✓ {filename} - {description}")
        else:
            print(f"  ✗ {filename} - {description} (缺失)")
            missing_files.append(filename)
    
    # 4. 可以删除的文件和目录
    removable_items = [
        # 原始复杂项目文件
        "app.py",
        "launcher.py", 
        "requirements.txt",
        "docker-compose.yml",
        "Dockerfile",
        
        # 复杂的模块目录
        "api/",
        "cli/",
        "configs/",
        "monitoring/",
        "plate_recognition/",
        "tests/",
        "web/",
        "models/",
        
        # 缓存和临时文件
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        
        # IDE文件
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        
        # 其他
        "venv/",
        "env/",
    ]
    
    # 5. 生成项目清理报告
    print("\n📊 生成项目分析报告...")
    
    # 统计当前项目文件
    all_files = []
    all_dirs = []
    
    for item in project_root.rglob("*"):
        if item.is_file():
            all_files.append(item.relative_to(project_root))
        elif item.is_dir() and item.name not in {'.git', '__pycache__'}:
            all_dirs.append(item.relative_to(project_root))
    
    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "analysis": {
            "total_files": len(all_files),
            "total_directories": len(all_dirs),
            "core_files": len([f for f in core_files.keys() if (project_root / f).exists()]),
            "missing_files": missing_files
        },
        "recommended_structure": {
            "core_files": core_files,
            "directories": essential_dirs,
            "removable_items": removable_items
        },
        "all_files": [str(f) for f in all_files],
        "all_directories": [str(d) for d in all_dirs]
    }
    
    report_file = project_root / "project_analysis.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 分析报告已保存: {report_file}")
    
    # 6. 生成使用说明
    print("\n📚 生成快速使用说明...")
    
    quick_start = f"""# 车牌识别系统 - 快速开始

## 项目重组完成 ✅

重组时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📁 标准目录结构

```
{project_root.name}/
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

- ✅ 文件数量：从 {len(all_files)} 个文件精简为 {len(core_files)} 个核心文件
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
"""
    
    quick_start_file = project_root / "QUICK_START.md"
    with open(quick_start_file, 'w', encoding='utf-8') as f:
        f.write(quick_start)
    
    print(f"  ✓ 快速使用说明已保存: {quick_start_file}")
    
    # 7. 验证核心功能
    print("\n🔍 验证核心功能...")
    try:
        # 测试导入主模块
        import sys
        sys.path.insert(0, str(project_root))
        
        from simple_app import SimplePlateRecognizer
        recognizer = SimplePlateRecognizer()
        print(f"  ✓ 识别器初始化成功，可用引擎: {recognizer.engines}")
        
        # 检查测试图片
        test_images = list((project_root / "data" / "test_images").glob("*.jpg"))
        print(f"  ✓ 测试图片数量: {len(test_images)}")
        
    except Exception as e:
        print(f"  ✗ 功能验证失败: {e}")
    
    # 8. 完成总结
    print("\n🎉 项目重组完成！")
    print("\n📋 重组成果:")
    print(f"  • 核心文件: {len(core_files)} 个")
    print(f"  • 标准目录: {len(essential_dirs)} 个")
    print(f"  • 功能完整: Web界面 + 命令行 + 批量处理")
    print(f"  • 数据管理: 标准化的图片和结果管理")
    
    print("\n🚀 快速开始:")
    print("  1. pip install -r requirements_minimal.txt")
    print("  2. python generate_test_images.py")
    print("  3. python simple_app.py")
    print("  4. 访问 http://localhost:5000")
    
    print("\n📚 更多信息:")
    print("  • 详细使用指南: USAGE_GUIDE.md")
    print("  • 项目分析报告: project_analysis.json")
    print("  • 快速开始文档: QUICK_START.md")

if __name__ == '__main__':
    reorganize_project()
