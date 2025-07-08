# -*- coding: utf-8 -*-
"""
高级车牌检测模块 - 车牌识别系统
使用多种算法进行车牌定位和检测，提高检测准确率
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from config import Config

class AdvancedPlateDetector:
    """高级车牌检测器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        
    def detect_plates(self, image: np.ndarray, method: str = "comprehensive") -> List[Dict[str, Any]]:
        """
        综合车牌检测
        
        Args:
            image: 输入图像
            method: 检测方法 ("contour", "color", "cascade", "comprehensive")
            
        Returns:
            List[Dict]: 检测到的车牌信息列表
        """
        try:
            results = []
            
            if method == "contour" or method == "comprehensive":
                contour_results = self._detect_by_contour(image)
                results.extend(contour_results)
            
            if method == "color" or method == "comprehensive":
                color_results = self._detect_by_color(image)
                results.extend(color_results)
            
            if method == "comprehensive":
                # 融合多种方法的结果
                results = self._merge_detections(results)
            
            # 按置信度排序
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"车牌检测失败: {e}")
            return []
    
    def _detect_by_contour(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """基于轮廓的车牌检测"""
        results = []
        
        try:
            # 预处理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # 多种边缘检测方法
            edge_methods = [
                ('canny', lambda x: cv2.Canny(x, 50, 150)),
                ('canny_strict', lambda x: cv2.Canny(x, 100, 200)),
                ('canny_loose', lambda x: cv2.Canny(x, 30, 100))
            ]
            
            for method_name, edge_func in edge_methods:
                edges = edge_func(gray)
                
                # 形态学处理连接断开的边缘
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                
                # 查找轮廓
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 筛选车牌候选区域
                for contour in contours:
                    plate_info = self._analyze_contour(contour, image, method_name)
                    if plate_info:
                        results.append(plate_info)
            
            return results
            
        except Exception as e:
            self.logger.error(f"轮廓检测失败: {e}")
            return []
    
    def _detect_by_color(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """基于颜色的车牌检测"""
        results = []
        
        try:
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 检测不同颜色的车牌
            color_ranges = {
                'blue': self.config.BLUE_PLATE_HSV_RANGE,
                'green': self.config.GREEN_PLATE_HSV_RANGE,
                'yellow': self.config.YELLOW_PLATE_HSV_RANGE
            }
            
            for color_name, color_range in color_ranges.items():
                # 创建颜色掩码
                lower = np.array(color_range['lower'])
                upper = np.array(color_range['upper'])
                mask = cv2.inRange(hsv, lower, upper)
                
                # 形态学处理
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # 查找轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    plate_info = self._analyze_contour(contour, image, f"color_{color_name}")
                    if plate_info:
                        plate_info['color'] = color_name
                        results.append(plate_info)
            
            return results
            
        except Exception as e:
            self.logger.error(f"颜色检测失败: {e}")
            return []
    
    def _analyze_contour(self, contour: np.ndarray, image: np.ndarray, method: str) -> Optional[Dict[str, Any]]:
        """分析轮廓是否为车牌候选区域"""
        try:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < self.config.PLATE_MIN_AREA or area > self.config.PLATE_MAX_AREA:
                return None
            
            # 获取边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 检查宽高比
            aspect_ratio = w / h
            if not (self.config.PLATE_ASPECT_RATIO_MIN <= aspect_ratio <= self.config.PLATE_ASPECT_RATIO_MAX):
                return None
            
            # 检查矩形度（轮廓面积与边界矩形面积的比值）
            rect_area = w * h
            extent = area / rect_area
            if extent < 0.5:  # 矩形度太低
                return None
            
            # 裁剪车牌区域
            plate_region = image[y:y+h, x:x+w]
            
            # 计算置信度
            confidence = self._calculate_confidence(contour, plate_region, method)
            
            return {
                'bbox': (x, y, w, h),
                'region': plate_region,
                'contour': contour,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'confidence': confidence,
                'method': method
            }
            
        except Exception as e:
            self.logger.error(f"轮廓分析失败: {e}")
            return None
    
    def _calculate_confidence(self, contour: np.ndarray, plate_region: np.ndarray, method: str) -> float:
        """计算检测置信度"""
        try:
            confidence = 0.0
            
            # 基础分数
            if method.startswith('color'):
                confidence += 0.3  # 颜色检测基础分
            else:
                confidence += 0.2  # 轮廓检测基础分
            
            # 形状分析
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)
            
            if hull_area > 0:
                solidity = contour_area / hull_area
                if solidity > 0.8:  # 高凸性
                    confidence += 0.2
                elif solidity > 0.6:
                    confidence += 0.1
            
            # 边缘密度分析
            if plate_region is not None and plate_region.size > 0:
                gray_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY) if len(plate_region.shape) == 3 else plate_region
                edges = cv2.Canny(gray_region, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                if edge_density > 0.1:  # 足够的边缘密度
                    confidence += 0.3
                elif edge_density > 0.05:
                    confidence += 0.1
            
            # 文本特征分析（简单版本）
            if plate_region is not None and plate_region.size > 0:
                # 检查是否有字符样式的连通区域
                gray_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY) if len(plate_region.shape) == 3 else plate_region
                _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 查找连通区域
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 统计可能的字符区域
                char_count = 0
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > 5 and h > 10 and 0.2 < w/h < 3:  # 字符宽高比
                        char_count += 1
                
                if char_count >= 5:  # 至少5个字符
                    confidence += 0.2
                elif char_count >= 3:
                    confidence += 0.1
            
            return min(confidence, 1.0)  # 限制在0-1之间
            
        except Exception as e:
            self.logger.error(f"置信度计算失败: {e}")
            return 0.5
    
    def _merge_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """融合多种检测方法的结果"""
        if not detections:
            return []
        
        merged_results = []
        used_indices = set()
        
        for i, detection1 in enumerate(detections):
            if i in used_indices:
                continue
                
            # 查找重叠的检测结果
            overlapping_detections = [detection1]
            used_indices.add(i)
            
            bbox1 = detection1['bbox']
            for j, detection2 in enumerate(detections[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                bbox2 = detection2['bbox']
                
                # 计算IoU（交并比）
                iou = self._calculate_iou(bbox1, bbox2)
                if iou > 0.5:  # 重叠度超过50%
                    overlapping_detections.append(detection2)
                    used_indices.add(j)
            
            # 融合重叠的检测结果
            if len(overlapping_detections) > 1:
                merged_detection = self._merge_overlapping_detections(overlapping_detections)
                merged_results.append(merged_detection)
            else:
                merged_results.append(detection1)
        
        return merged_results
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集
        x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection_area = x_intersection * y_intersection
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0
        
        return intersection_area / union_area
    
    def _merge_overlapping_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """融合重叠的检测结果"""
        # 选择置信度最高的作为基础
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        # 计算平均置信度
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
        
        # 融合方法信息
        methods = [d['method'] for d in detections]
        merged_method = f"merged({','.join(set(methods))})"
        
        # 创建融合结果
        merged_result = best_detection.copy()
        merged_result['confidence'] = avg_confidence
        merged_result['method'] = merged_method
        merged_result['fusion_count'] = len(detections)
        
        return merged_result
    
    def refine_plate_region(self, plate_region: np.ndarray) -> np.ndarray:
        """精细化车牌区域"""
        try:
            if plate_region is None or plate_region.size == 0:
                return plate_region
            
            # 转换为灰度图
            if len(plate_region.shape) == 3:
                gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_region.copy()
            
            # 去除边框
            h, w = gray.shape
            border_size = max(2, min(h, w) // 20)
            
            # 裁剪边框
            refined = gray[border_size:h-border_size, border_size:w-border_size]
            
            # 如果裁剪后太小，则返回原图
            if refined.shape[0] < 20 or refined.shape[1] < 60:
                return plate_region
            
            return refined
            
        except Exception as e:
            self.logger.error(f"车牌区域精细化失败: {e}")
            return plate_region
    
    def validate_plate_region(self, plate_region: np.ndarray) -> Dict[str, Any]:
        """验证车牌区域的质量"""
        try:
            validation_result = {
                'is_valid': False,
                'scores': {},
                'total_score': 0.0,
                'issues': []
            }
            
            if plate_region is None or plate_region.size == 0:
                validation_result['issues'].append("图像为空")
                return validation_result
            
            # 尺寸检查
            h, w = plate_region.shape[:2]
            if h < 20 or w < 60:
                validation_result['issues'].append("尺寸过小")
                validation_result['scores']['size'] = 0.0
            else:
                # 宽高比检查
                aspect_ratio = w / h
                if 2.0 <= aspect_ratio <= 6.0:
                    validation_result['scores']['aspect_ratio'] = 1.0
                else:
                    validation_result['scores']['aspect_ratio'] = 0.5
                    validation_result['issues'].append(f"宽高比异常: {aspect_ratio:.2f}")
            
            # 对比度检查
            if len(plate_region.shape) == 3:
                gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_region
            
            contrast = np.std(gray)
            if contrast > 30:
                validation_result['scores']['contrast'] = 1.0
            elif contrast > 15:
                validation_result['scores']['contrast'] = 0.7
            else:
                validation_result['scores']['contrast'] = 0.3
                validation_result['issues'].append(f"对比度低: {contrast:.2f}")
            
            # 边缘密度检查
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            if edge_density > 0.1:
                validation_result['scores']['edge_density'] = 1.0
            elif edge_density > 0.05:
                validation_result['scores']['edge_density'] = 0.7
            else:
                validation_result['scores']['edge_density'] = 0.3
                validation_result['issues'].append(f"边缘密度低: {edge_density:.3f}")
            
            # 计算总分
            if validation_result['scores']:
                validation_result['total_score'] = sum(validation_result['scores'].values()) / len(validation_result['scores'])
                validation_result['is_valid'] = validation_result['total_score'] > 0.6
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"车牌区域验证失败: {e}")
            return {'is_valid': False, 'scores': {}, 'total_score': 0.0, 'issues': [str(e)]}
