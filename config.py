# -*- coding: utf-8 -*-
"""
配置文件 - 车牌识别系统
包含所有系统配置参数和常量定义
"""

import os
from typing import Dict, List, Tuple

class Config:
    """系统配置类"""
    
    # === 基础配置 ===
    DEBUG = True
    LOG_LEVEL = "INFO"
    LOG_FILE = "license_plate_recognition.log"
    
    # === API配置 ===
    GEMINI_API_KEY = "AIzaSyARL4h588FeWT-eSUdQPqfZeWcmifLDjb0"  # 请替换为您的API密钥
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5000
    
    # === OCR引擎配置 ===
    TESSERACT_CMD = r'E:\application\PDF24\tesseract\tesseract.exe'  # 请根据实际路径修改
    TESSERACT_CONFIG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
    
    # === PaddleOCR配置 ===
    PADDLE_USE_ANGLE_CLS = True
    PADDLE_LANG = 'ch'
    PADDLE_USE_GPU = False  # 如果有GPU可设为True
    
    # === 图像处理参数 ===
    IMAGE_MAX_SIZE = (1920, 1080)
    PLATE_MIN_AREA = 500
    PLATE_MAX_AREA = 50000
    PLATE_ASPECT_RATIO_MIN = 1.5
    PLATE_ASPECT_RATIO_MAX = 8.0
    
    # === 车牌格式配置 ===
    PROVINCES = ['京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘', '皖', 
                '鲁', '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕', 
                '吉', '闽', '贵', '粤', '青', '藏', '川', '宁', '琼']
    
    PLATE_PATTERN = r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][0-9A-Z]{4,6}$'
    
    # === CRNN模型配置 ===
    MODEL_INPUT_SIZE = (120, 32)  # 宽x高
    MODEL_CHANNELS = 3
    NUM_CLASSES = 68  # 车牌字符类别数
    
    # === 数据增强参数 ===
    AUGMENTATION_CONFIG = {
        'rotation_range': 10,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'shear_range': 0.1,
        'zoom_range': 0.1,
        'brightness_range': (0.8, 1.2),
        'noise_factor': 0.05
    }
    
    # === 性能监控配置 ===
    MONITORING_ENABLED = True
    PROMETHEUS_PORT = 8000
    MAX_REQUEST_TIME = 30  # 秒
    
    # === 文件路径配置 ===
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    TEMP_DIR = os.path.join(BASE_DIR, "temp")
    TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
    
    # === 模型文件路径 ===
    CRNN_MODEL_PATH = os.path.join(MODELS_DIR, "crnn_plate_model.h5")
    DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "plate_detector.xml")
    
    # === 颜色范围配置（HSV） ===
    BLUE_PLATE_HSV_RANGE = {
        'lower': (100, 50, 50),
        'upper': (130, 255, 255)
    }
    
    GREEN_PLATE_HSV_RANGE = {
        'lower': (40, 50, 50),
        'upper': (80, 255, 255)
    }
    
    YELLOW_PLATE_HSV_RANGE = {
        'lower': (20, 50, 50),
        'upper': (30, 255, 255)
    }
    
    # === 识别置信度阈值 ===
    CONFIDENCE_THRESHOLD = {
        'tesseract': 60,
        'paddle': 0.8,
        'gemini': 0.7,
        'crnn': 0.85
    }
    
    # === 系统性能参数 ===
    MAX_CONCURRENT_REQUESTS = 10
    CACHE_SIZE = 100
    REQUEST_TIMEOUT = 30
    
    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        for directory in [cls.MODELS_DIR, cls.LOGS_DIR, cls.TEMP_DIR, cls.TEST_DATA_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_character_set(cls) -> List[str]:
        """获取车牌字符集"""
        chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        chars.extend(cls.PROVINCES)
        return chars

# 错误码定义
class ErrorCodes:
    """错误码定义"""
    SUCCESS = 0
    IMAGE_LOAD_ERROR = 1001
    PLATE_NOT_FOUND = 1002
    OCR_ERROR = 1003
    MODEL_ERROR = 1004
    API_ERROR = 1005
    VALIDATION_ERROR = 1006
    TIMEOUT_ERROR = 1007
    
    @classmethod
    def get_error_message(cls, code: int) -> str:
        """获取错误信息"""
        error_messages = {
            cls.SUCCESS: "操作成功",
            cls.IMAGE_LOAD_ERROR: "图像加载失败",
            cls.PLATE_NOT_FOUND: "未检测到车牌",
            cls.OCR_ERROR: "OCR识别失败",
            cls.MODEL_ERROR: "模型预测失败",
            cls.API_ERROR: "API调用失败",
            cls.VALIDATION_ERROR: "数据验证失败",
            cls.TIMEOUT_ERROR: "请求超时"
        }
        return error_messages.get(code, "未知错误")

# 初始化配置
Config.ensure_directories()
