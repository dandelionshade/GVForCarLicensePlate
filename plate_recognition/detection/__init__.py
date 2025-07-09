# -*- coding: utf-8 -*-
"""
检测模块初始化文件
"""

from .detector import (
    BaseDetector,
    ContourDetector,
    ColorDetector,
    PlateDetector
)

__all__ = [
    'BaseDetector',
    'ContourDetector', 
    'ColorDetector',
    'PlateDetector'
]
