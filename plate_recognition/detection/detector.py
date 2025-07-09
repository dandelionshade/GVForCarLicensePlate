# -*- coding: utf-8 -*-
"""
车牌检测模块 - 基础检测器接口和实现

这个模块包含了车牌检测的核心功能，将原来的plate_detector.py
重构为更加模块化的检测系统
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import logging

from ..core.config import get_config
from ..core.exceptions import DetectionError, ErrorCodes
from ..core.constants import PlateConstants, DetectionMethod


class BaseDetector(ABC):
    """车牌检测器基类"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """检测车牌
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果列表，每个结果包含：
            - bbox: 边界框 (x, y, w, h)
            - confidence: 置信度
            - region: 车牌区域图像
            - method: 检测方法
        """
        pass
    
    def validate_detection(self, bbox: Tuple[int, int, int, int], 
                          image_shape: Tuple[int, int]) -> bool:
        """验证检测结果是否有效"""
        x, y, w, h = bbox
        
        # 检查边界
        if x < 0 or y < 0 or x + w > image_shape[1] or y + h > image_shape[0]:
            return False
        
        # 检查尺寸
        area = w * h
        if not PlateConstants.is_valid_area(area):
            return False
        
        # 检查宽高比
        if not PlateConstants.is_valid_aspect_ratio(w, h):
            return False
        
        return True
    
    def extract_region(self, image: np.ndarray, 
                      bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """提取车牌区域"""
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]


class ContourDetector(BaseDetector):
    """基于轮廓的车牌检测器"""
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用轮廓检测车牌"""
        try:
            results = []
            
            # 预处理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 边缘检测
            edges = cv2.Canny(blurred, 50, 150)
            
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
            morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, w, h)
                
                # 验证检测结果
                if not self.validate_detection(bbox, image.shape[:2]):
                    continue
                
                # 计算置信度
                confidence = self._calculate_contour_confidence(contour, w, h)
                
                # 提取区域
                region = self.extract_region(image, bbox)
                
                results.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'region': region,
                    'method': DetectionMethod.CONTOUR.value,
                    'area': w * h,
                    'aspect_ratio': w / h if h > 0 else 0
                })
            
            # 按置信度排序
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results
            
        except Exception as e:
            self.logger.error(f"轮廓检测失败: {e}")
            raise DetectionError(f"轮廓检测失败: {e}", ErrorCodes.DETECTION_ERROR)
    
    def _calculate_contour_confidence(self, contour: np.ndarray, 
                                    width: int, height: int) -> float:
        """计算轮廓检测的置信度"""
        try:
            # 计算轮廓面积与边界框面积的比值
            contour_area = cv2.contourArea(contour)
            bbox_area = width * height
            
            if bbox_area == 0:
                return 0.0
            
            area_ratio = contour_area / bbox_area
            
            # 计算宽高比评分
            aspect_ratio = width / height if height > 0 else 0
            aspect_score = 0.0
            
            if 2.0 <= aspect_ratio <= 8.0:
                # 理想宽高比范围内给高分
                if 3.0 <= aspect_ratio <= 5.0:
                    aspect_score = 1.0
                else:
                    aspect_score = 0.7
            
            # 综合置信度
            confidence = (area_ratio * 0.6 + aspect_score * 0.4)
            return min(confidence, 1.0)
            
        except Exception:
            return 0.0


class ColorDetector(BaseDetector):
    """基于颜色的车牌检测器"""
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用颜色检测车牌"""
        try:
            results = []
            
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 检测不同颜色的车牌
            for color_name in ['blue', 'yellow', 'green', 'white']:
                color_results = self._detect_by_color(image, hsv, color_name)
                results.extend(color_results)
            
            # 去重和排序
            results = self._remove_duplicates(results)
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"颜色检测失败: {e}")
            raise DetectionError(f"颜色检测失败: {e}", ErrorCodes.DETECTION_ERROR)
    
    def _detect_by_color(self, image: np.ndarray, hsv: np.ndarray, 
                        color_name: str) -> List[Dict[str, Any]]:
        """检测特定颜色的车牌"""
        results = []
        
        try:
            # 获取颜色范围
            color_ranges = PlateConstants.COLOR_RANGES.get(color_name)
            if not color_ranges:
                return results
            
            # 创建颜色掩码
            lower = np.array(color_ranges['lower'])
            upper = np.array(color_ranges['upper'])
            mask = cv2.inRange(hsv, lower, upper)
            
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, w, h)
                
                # 验证检测结果
                if not self.validate_detection(bbox, image.shape[:2]):
                    continue
                
                # 计算置信度
                confidence = self._calculate_color_confidence(mask, bbox, color_name)
                
                # 提取区域
                region = self.extract_region(image, bbox)
                
                results.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'region': region,
                    'method': DetectionMethod.COLOR.value,
                    'color': color_name,
                    'area': w * h,
                    'aspect_ratio': w / h if h > 0 else 0
                })
        
        except Exception as e:
            self.logger.warning(f"检测颜色 {color_name} 失败: {e}")
        
        return results
    
    def _calculate_color_confidence(self, mask: np.ndarray, 
                                   bbox: Tuple[int, int, int, int], 
                                   color_name: str) -> float:
        """计算颜色检测的置信度"""
        try:
            x, y, w, h = bbox
            
            # 提取掩码区域
            roi_mask = mask[y:y+h, x:x+w]
            
            # 计算颜色覆盖率
            total_pixels = w * h
            color_pixels = cv2.countNonZero(roi_mask)
            
            if total_pixels == 0:
                return 0.0
            
            coverage_ratio = color_pixels / total_pixels
            
            # 根据颜色类型调整基础置信度
            base_confidence = {
                'blue': 0.8,    # 蓝牌最常见
                'yellow': 0.7,  # 黄牌次之
                'green': 0.6,   # 绿牌新能源
                'white': 0.5    # 白牌较少见
            }.get(color_name, 0.5)
            
            # 计算最终置信度
            confidence = coverage_ratio * base_confidence
            return min(confidence, 1.0)
            
        except Exception:
            return 0.0
    
    def _remove_duplicates(self, results: List[Dict[str, Any]], 
                          overlap_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """去除重复检测结果"""
        if len(results) <= 1:
            return results
        
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_results = []
        
        for result in results:
            is_duplicate = False
            current_bbox = result['bbox']
            
            for existing in filtered_results:
                existing_bbox = existing['bbox']
                
                # 计算重叠率
                overlap = self._calculate_overlap(current_bbox, existing_bbox)
                
                if overlap > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_results.append(result)
        
        return filtered_results
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的重叠率"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union


class PlateDetector:
    """车牌检测器主类 - 整合多种检测方法"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化各种检测器
        self.contour_detector = ContourDetector()
        self.color_detector = ColorDetector()
    
    def detect_plates(self, image: np.ndarray, 
                     method: str = "comprehensive") -> List[Dict[str, Any]]:
        """检测车牌
        
        Args:
            image: 输入图像
            method: 检测方法 ('contour', 'color', 'comprehensive')
            
        Returns:
            检测结果列表
        """
        try:
            if method == DetectionMethod.CONTOUR.value:
                return self.contour_detector.detect(image)
            elif method == DetectionMethod.COLOR.value:
                return self.color_detector.detect(image)
            elif method == DetectionMethod.COMPREHENSIVE.value:
                return self._comprehensive_detect(image)
            else:
                raise DetectionError(f"不支持的检测方法: {method}", ErrorCodes.DETECTION_ERROR)
        
        except Exception as e:
            self.logger.error(f"车牌检测失败: {e}")
            if isinstance(e, DetectionError):
                raise
            else:
                raise DetectionError(f"车牌检测失败: {e}", ErrorCodes.DETECTION_ERROR)
    
    def _comprehensive_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """综合检测方法 - 融合多种检测结果"""
        all_results = []
        
        try:
            # 轮廓检测
            contour_results = self.contour_detector.detect(image)
            all_results.extend(contour_results)
            
            # 颜色检测
            color_results = self.color_detector.detect(image)
            all_results.extend(color_results)
            
            # 去重和融合
            final_results = self._merge_results(all_results)
            
            # 排序
            final_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"综合检测失败: {e}")
            return all_results  # 返回部分结果
    
    def _merge_results(self, results: List[Dict[str, Any]], 
                      overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """合并检测结果"""
        if len(results) <= 1:
            return results
        
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged_results = []
        
        for result in results:
            is_merged = False
            current_bbox = result['bbox']
            
            for i, existing in enumerate(merged_results):
                existing_bbox = existing['bbox']
                
                # 计算重叠率
                overlap = self.color_detector._calculate_overlap(current_bbox, existing_bbox)
                
                if overlap > overlap_threshold:
                    # 合并结果，选择置信度更高的
                    if result['confidence'] > existing['confidence']:
                        merged_results[i] = result
                    is_merged = True
                    break
            
            if not is_merged:
                merged_results.append(result)
        
        return merged_results
    
    def validate_plate_region(self, region: np.ndarray) -> Dict[str, Any]:
        """验证车牌区域质量"""
        validation_result = {
            'is_valid': False,
            'score': 0.0,
            'issues': []
        }
        
        try:
            if region is None or region.size == 0:
                validation_result['issues'].append('区域为空')
                return validation_result
            
            height, width = region.shape[:2]
            
            # 检查尺寸
            if width < 80 or height < 20:
                validation_result['issues'].append('尺寸过小')
            
            # 检查宽高比
            aspect_ratio = width / height if height > 0 else 0
            if not PlateConstants.is_valid_aspect_ratio(width, height):
                validation_result['issues'].append(f'宽高比异常: {aspect_ratio:.2f}')
            
            # 检查图像质量
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            
            # 计算对比度
            contrast = np.std(gray)
            if contrast < 20:
                validation_result['issues'].append('对比度过低')
            
            # 计算清晰度（拉普拉斯算子）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                validation_result['issues'].append('图像模糊')
            
            # 计算总评分
            score = 0.0
            if len(validation_result['issues']) == 0:
                score = 1.0
            elif len(validation_result['issues']) <= 2:
                score = 0.7
            else:
                score = 0.3
            
            validation_result['score'] = score
            validation_result['is_valid'] = score >= 0.5
            
        except Exception as e:
            validation_result['issues'].append(f'验证失败: {e}')
        
        return validation_result
