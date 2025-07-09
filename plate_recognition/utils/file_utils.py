# -*- coding: utf-8 -*-
"""
文件工具模块
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def clean_filename(filename):
    """清理文件名"""
    import re
    # 移除或替换不安全的字符
    cleaned = re.sub(r'[<>:"/\|?*]', '_', filename)
    return cleaned

def get_file_size(filepath):
    """获取文件大小（字节）"""
    return Path(filepath).stat().st_size

def get_files_by_extension(directory, extensions):
    """根据扩展名获取文件列表"""
    directory = Path(directory)
    files = []
    
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(files)

def safe_copy(src, dst, overwrite=False):
    """安全复制文件"""
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")
    
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在: {dst}")
    
    # 确保目标目录存在
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(src_path, dst_path)
    return dst_path
