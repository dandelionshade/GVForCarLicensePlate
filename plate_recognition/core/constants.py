# -*- coding: utf-8 -*-
"""
系统常量定义 - 车牌识别系统

定义车牌识别相关的常量和枚举值
"""

import re
from enum import Enum
from typing import List, Tuple, Dict


class PlateType(Enum):
    """车牌类型枚举"""
    STANDARD = "standard"        # 标准车牌
    NEW_ENERGY = "new_energy"    # 新能源车牌
    MILITARY = "military"        # 军用车牌
    POLICE = "police"           # 警用车牌
    EMBASSY = "embassy"         # 使馆车牌


class PlateColor(Enum):
    """车牌颜色枚举"""
    BLUE = "blue"      # 蓝牌
    YELLOW = "yellow"  # 黄牌  
    GREEN = "green"    # 绿牌（新能源）
    WHITE = "white"    # 白牌
    BLACK = "black"    # 黑牌


class OCREngine(Enum):
    """OCR引擎枚举"""
    TESSERACT = "tesseract"
    PADDLE = "paddle"
    GEMINI = "gemini"
    CRNN = "crnn"


class DetectionMethod(Enum):
    """检测方法枚举"""
    CONTOUR = "contour"          # 轮廓检测
    COLOR = "color"              # 颜色检测
    EDGE = "edge"                # 边缘检测
    COMPREHENSIVE = "comprehensive"  # 综合检测


class PlateConstants:
    """车牌相关常量"""
    
    # 省份简称
    PROVINCES: List[str] = [
        '京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘', '皖', 
        '鲁', '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕', 
        '吉', '闽', '贵', '粤', '青', '藏', '川', '宁', '琼'
    ]
    
    # 字母表（不包含I和O）
    LETTERS: List[str] = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 
        'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
    
    # 数字
    DIGITS: List[str] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # 车牌格式正则表达式
    PLATE_PATTERNS: Dict[str, str] = {
        'standard': r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][0-9A-Z]{5}$',
        'new_energy': r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][0-9A-Z]{5,6}$',
        'short_truck': r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][0-9A-Z]{4}$'
    }
    
    # 车牌尺寸相关常量
    PLATE_DIMENSIONS: Dict[str, Tuple[int, int]] = {
        'standard': (440, 140),      # 标准车牌尺寸 (mm)
        'new_energy': (480, 140),    # 新能源车牌尺寸 (mm)
        'motorcycle': (220, 140)     # 摩托车车牌尺寸 (mm)
    }
    
    # 车牌宽高比范围
    ASPECT_RATIO_RANGE: Tuple[float, float] = (2.0, 8.0)
    
    # 车牌区域面积范围（像素）
    AREA_RANGE: Tuple[int, int] = (500, 50000)
    
    # 颜色阈值（HSV）
    COLOR_RANGES: Dict[str, Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
        'blue': {
            'lower': (100, 50, 50),
            'upper': (130, 255, 255)
        },
        'yellow': {
            'lower': (20, 50, 50),
            'upper': (30, 255, 255)
        },
        'green': {
            'lower': (40, 50, 50),
            'upper': (80, 255, 255)
        },
        'white': {
            'lower': (0, 0, 200),
            'upper': (180, 30, 255)
        }
    }
    
    # OCR引擎置信度阈值
    CONFIDENCE_THRESHOLDS: Dict[str, float] = {
        'tesseract': 0.6,
        'paddle': 0.7,
        'gemini': 0.8,
        'crnn': 0.75
    }
    
    # 图像预处理参数
    PREPROCESSING_PARAMS: Dict[str, Dict] = {
        'gaussian_blur': {'kernel_size': (5, 5), 'sigma': 0},
        'bilateral_filter': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
        'morphology': {'kernel_size': (3, 3), 'iterations': 1},
        'resize': {'target_height': 32, 'target_width': 120}
    }
    
    @classmethod
    def validate_plate_number(cls, plate_number: str, plate_type: str = 'standard') -> bool:
        """验证车牌号码格式"""
        if not plate_number or len(plate_number) < 7:
            return False
        
        pattern = cls.PLATE_PATTERNS.get(plate_type, cls.PLATE_PATTERNS['standard'])
        return bool(re.match(pattern, plate_number))
    
    @classmethod
    def get_province_from_plate(cls, plate_number: str) -> str:
        """从车牌号码获取省份"""
        if plate_number and len(plate_number) > 0:
            return plate_number[0]
        return ""
    
    @classmethod
    def get_city_code_from_plate(cls, plate_number: str) -> str:
        """从车牌号码获取城市代码"""
        if plate_number and len(plate_number) > 1:
            return plate_number[1]
        return ""
    
    @classmethod
    def is_valid_aspect_ratio(cls, width: float, height: float) -> bool:
        """检查宽高比是否在有效范围内"""
        if height == 0:
            return False
        ratio = width / height
        return cls.ASPECT_RATIO_RANGE[0] <= ratio <= cls.ASPECT_RATIO_RANGE[1]
    
    @classmethod
    def is_valid_area(cls, area: int) -> bool:
        """检查面积是否在有效范围内"""
        return cls.AREA_RANGE[0] <= area <= cls.AREA_RANGE[1]
