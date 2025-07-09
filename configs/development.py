# -*- coding: utf-8 -*-
"""
开发环境配置
"""

import os
from pathlib import Path

# 基础配置
DEBUG = True
TESTING = False

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'
TEMP_DIR = PROJECT_ROOT / 'temp'

# Web配置
WEB_HOST = '127.0.0.1'
WEB_PORT = 5000
UPLOAD_FOLDER = PROJECT_ROOT / 'web' / 'static' / 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# API配置
API_HOST = '127.0.0.1'
API_PORT = 8000
API_WORKERS = 1

# 识别配置
DEFAULT_METHOD = 'fusion'
ENABLE_GPU = False
BATCH_SIZE = 1
CONFIDENCE_THRESHOLD = 0.5

# OCR配置
TESSERACT_PATH = r'E:\application\PDF24\tesseract\tesseract.exe'
TESSERACT_CONFIG = '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领'

# PaddleOCR配置
PADDLEOCR_USE_GPU = False
PADDLEOCR_LANG = 'ch'

# Gemini API配置
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyARL4h588FeWT-eSUdQPqfZeWcmifLDjb0')
GEMINI_MODEL = 'gemini-1.5-flash'

# 监控配置
MONITORING_ENABLED = True
METRICS_RETENTION_DAYS = 30
PERFORMANCE_LOG_LEVEL = 'INFO'

# 确保必要目录存在
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, TEMP_DIR, UPLOAD_FOLDER]:
    directory.mkdir(parents=True, exist_ok=True)
