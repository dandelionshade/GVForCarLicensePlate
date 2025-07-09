#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目清理脚本
删除不必要的文件，保留最小可行版本
"""

import os
import shutil
from pathlib import Path

def clean_project():
    """清理项目，保留核心文件"""
    
    project_root = Path(__file__).parent
    
    # 要保留的文件和目录
    keep_files = {
        'simple_app.py',
        'cli_simple.py', 
        'requirements_minimal.txt',
        'README_minimal.md',
        'clean_project.py',
        'data/',  # 可能包含测试数据
        'logs/',  # 日志目录
        '.git/',  # Git仓库
        '.gitignore',
        'README.md'  # 原始README
    }
    
    # 要删除的目录
    delete_dirs = [
        'api/',
        'cli/',
        'configs/',
        'monitoring/',
        'plate_recognition/',
        'tests/',
        'web/',
        'models/',
        '__pycache__/',
        '*.pyc'
    ]
    
    # 要删除的文件
    delete_files = [
        'app.py',
        'launcher.py',
        'requirements.txt',
        'docker-compose.yml',
        'Dockerfile'
    ]
    
    print("开始清理项目...")
    
    # 删除目录
    for dir_pattern in delete_dirs:
        if dir_pattern.endswith('/'):
            dir_name = dir_pattern[:-1]
            dir_path = project_root / dir_name
            if dir_path.exists():
                print(f"删除目录: {dir_path}")
                shutil.rmtree(dir_path)
        else:
            # 处理通配符模式
            for path in project_root.rglob(dir_pattern):
                if path.is_file():
                    print(f"删除文件: {path}")
                    path.unlink()
                elif path.is_dir():
                    print(f"删除目录: {path}")
                    shutil.rmtree(path)
    
    # 删除文件
    for file_name in delete_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"删除文件: {file_path}")
            file_path.unlink()
    
    # 创建必要的目录
    (project_root / 'data').mkdir(exist_ok=True)
    (project_root / 'logs').mkdir(exist_ok=True)
    
    print("\n清理完成!")
    print("\n保留的文件结构:")
    print("├── simple_app.py           # 主应用文件")
    print("├── cli_simple.py          # 命令行工具")
    print("├── requirements_minimal.txt # 最小依赖")
    print("├── README_minimal.md      # 说明文档")
    print("├── clean_project.py       # 清理脚本")
    print("├── data/                  # 数据目录")
    print("└── logs/                  # 日志目录")
    
    print("\n使用说明:")
    print("1. 安装依赖: pip install -r requirements_minimal.txt")
    print("2. 启动Web服务: python simple_app.py")
    print("3. 命令行使用: python cli_simple.py image.jpg")
    print("4. 查看详细文档: README_minimal.md")


if __name__ == '__main__':
    confirm = input("确定要清理项目吗？这将删除大量文件！(y/N): ")
    if confirm.lower() == 'y':
        clean_project()
    else:
        print("已取消清理操作")
