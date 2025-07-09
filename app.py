'''
Author: zhen doniajohary2677@gmail.com
Date: 2025-07-08 17:00:08
LastEditors: zhen doniajohary2677@gmail.com
LastEditTime: 2025-07-08 18:38:33
FilePath: \GVForCarLicensePlate\app.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
车牌识别系统 - Flask Web应用入口
提供Web界面和API接口，支持在线车牌识别
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import create_app

# 创建Flask应用实例
app = create_app()

if __name__ == '__main__':
    # 开发模式运行
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
