# -*- coding: utf-8 -*-
"""
车牌识别系统核心包

这个包包含了车牌识别系统的所有核心功能：
- 车牌检测
- OCR识别
- 图像预处理
- 深度学习模型
- 处理流水线
"""

__version__ = "2.0.0"
__author__ = "Your Name"

from .core.config import Config
from .core.exceptions import PlateRecognitionError
from .pipeline.recognition_pipeline import RecognitionPipeline

__all__ = [
    'Config',
    'PlateRecognitionError', 
    'RecognitionPipeline'
]
