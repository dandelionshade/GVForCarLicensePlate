# -*- coding: utf-8 -*-
"""
识别模块初始化文件
"""

from .multi_engine_ocr import (
    BaseOCR,
    TesseractOCR,
    PaddleOCR,
    GeminiOCR,
    MultiEngineOCR
)

__all__ = [
    'BaseOCR',
    'TesseractOCR',
    'PaddleOCR', 
    'GeminiOCR',
    'MultiEngineOCR'
]
