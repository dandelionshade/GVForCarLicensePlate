# -*- coding: utf-8 -*-
"""
图像预处理模块

将原来的image_processor.py重构为更加模块化的预处理系统
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

from ..core.config import get_config
from ..core.exceptions import ImageProcessingError, ErrorCodes
from ..core.constants import PlateConstants


class ImageProcessor:
    """图像预处理器"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def preprocess_image(self, image: np.ndarray, 
                        method: str = "comprehensive") -> Dict[str, np.ndarray]:
        """预处理图像
        
        Args:
            image: 输入图像
            method: 预处理方法 ('basic', 'denoising', 'enhancement', 'comprehensive')
            
        Returns:
            预处理结果字典，包含不同预处理方法的结果
        """
        try:
            results = {'original': image.copy()}
            
            if method == "basic":
                results['processed'] = self._basic_preprocess(image)
            elif method == "denoising":
                results['denoised'] = self._denoise_image(image)
            elif method == "enhancement":
                results['enhanced'] = self._enhance_image(image)
            elif method == "comprehensive":
                # 应用所有预处理方法
                results['denoised'] = self._denoise_image(image)
                results['enhanced'] = self._enhance_image(results['denoised'])
                results['sharpened'] = self._sharpen_image(results['enhanced'])
                results['normalized'] = self._normalize_image(results['sharpened'])
            else:
                raise ImageProcessingError(
                    f"不支持的预处理方法: {method}",
                    ErrorCodes.IMAGE_PROCESSING_ERROR
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            if isinstance(e, ImageProcessingError):
                raise
            else:
                raise ImageProcessingError(
                    f"图像预处理失败: {e}",
                    ErrorCodes.IMAGE_PROCESSING_ERROR
                )
    
    def _basic_preprocess(self, image: np.ndarray) -> np.ndarray:
        """基础预处理"""
        try:
            # 调整图像大小
            processed = self._resize_image(image)
            
            # 转换色彩空间
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            return processed
            
        except Exception as e:
            self.logger.warning(f"基础预处理失败: {e}")
            return image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """图像去噪"""
        try:
            if len(image.shape) == 3:
                # 彩色图像去噪
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                # 灰度图像去噪
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            
            return denoised
            
        except Exception as e:
            self.logger.warning(f"图像去噪失败: {e}")
            return image
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """图像增强"""
        try:
            # 转换为LAB色彩空间
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # 应用CLAHE到L通道
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # 合并通道
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                # 灰度图像直接应用CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"图像增强失败: {e}")
            return image
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """图像锐化"""
        try:
            # 定义锐化核
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            
            # 应用锐化
            sharpened = cv2.filter2D(image, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            self.logger.warning(f"图像锐化失败: {e}")
            return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """图像归一化"""
        try:
            # 归一化到0-255范围
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            
            return normalized.astype(np.uint8)
            
        except Exception as e:
            self.logger.warning(f"图像归一化失败: {e}")
            return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """调整图像大小"""
        try:
            height, width = image.shape[:2]
            max_width, max_height = self.config.image_processing.max_size
            
            # 检查是否需要调整大小
            if width <= max_width and height <= max_height:
                return image
            
            # 计算缩放比例
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 调整大小
            resized = cv2.resize(image, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA)
            
            return resized
            
        except Exception as e:
            self.logger.warning(f"图像大小调整失败: {e}")
            return image
    
    def enhance_plate_region(self, plate_region: np.ndarray) -> np.ndarray:
        """专门针对车牌区域的增强"""
        try:
            if plate_region is None or plate_region.size == 0:
                return plate_region
            
            # 1. 调整大小到标准尺寸
            target_height = 64
            height, width = plate_region.shape[:2]
            if height != target_height:
                scale = target_height / height
                new_width = int(width * scale)
                resized = cv2.resize(plate_region, (new_width, target_height),
                                   interpolation=cv2.INTER_CUBIC)
            else:
                resized = plate_region.copy()
            
            # 2. 转换为灰度图
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized.copy()
            
            # 3. 高斯模糊去噪
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # 4. 自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 2))
            enhanced = clahe.apply(blurred)
            
            # 5. 锐化
            kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 6. 形态学操作去除噪点
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morphed = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
            
            return morphed
            
        except Exception as e:
            self.logger.warning(f"车牌区域增强失败: {e}")
            return plate_region
    
    def correct_perspective(self, image: np.ndarray, 
                           corners: np.ndarray) -> Optional[np.ndarray]:
        """透视矫正"""
        try:
            if corners is None or len(corners) != 4:
                return None
            
            # 排序角点（左上、右上、右下、左下）
            corners = self._order_points(corners)
            
            # 计算目标矩形尺寸
            width = max(
                np.linalg.norm(corners[1] - corners[0]),  # 上边长度
                np.linalg.norm(corners[2] - corners[3])   # 下边长度
            )
            height = max(
                np.linalg.norm(corners[3] - corners[0]),  # 左边长度
                np.linalg.norm(corners[2] - corners[1])   # 右边长度
            )
            
            # 定义目标点
            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # 计算透视变换矩阵
            matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_points)
            
            # 应用透视变换
            corrected = cv2.warpPerspective(image, matrix, (int(width), int(height)))
            
            return corrected
            
        except Exception as e:
            self.logger.warning(f"透视矫正失败: {e}")
            return None
    
    def _order_points(self, points: np.ndarray) -> np.ndarray:
        """排序四个角点"""
        # 初始化排序后的坐标
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # 计算每个点的坐标和
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # 左上角（和最小）
        rect[2] = points[np.argmax(s)]  # 右下角（和最大）
        
        # 计算每个点的坐标差
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # 右上角（差最小）
        rect[3] = points[np.argmax(diff)]  # 左下角（差最大）
        
        return rect
    
    def analyze_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """分析图像质量"""
        try:
            quality_metrics = {}
            
            # 转换为灰度图进行分析
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 1. 清晰度（拉普拉斯算子方差）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics['sharpness'] = laplacian_var
            quality_metrics['is_sharp'] = laplacian_var > 100
            
            # 2. 对比度（标准差）
            contrast = np.std(gray)
            quality_metrics['contrast'] = contrast
            quality_metrics['has_good_contrast'] = contrast > 30
            
            # 3. 亮度
            brightness = np.mean(gray)
            quality_metrics['brightness'] = brightness
            quality_metrics['is_well_lit'] = 50 < brightness < 200
            
            # 4. 噪声级别（使用高频成分估计）
            kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])
            noise = cv2.filter2D(gray, -1, kernel)
            noise_level = np.std(noise)
            quality_metrics['noise_level'] = noise_level
            quality_metrics['is_low_noise'] = noise_level < 50
            
            # 5. 整体质量评分
            score = 0.0
            if quality_metrics['is_sharp']:
                score += 0.3
            if quality_metrics['has_good_contrast']:
                score += 0.3
            if quality_metrics['is_well_lit']:
                score += 0.2
            if quality_metrics['is_low_noise']:
                score += 0.2
            
            quality_metrics['overall_score'] = score
            quality_metrics['quality_level'] = self._get_quality_level(score)
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"图像质量分析失败: {e}")
            return {'error': str(e)}
    
    def _get_quality_level(self, score: float) -> str:
        """根据评分获取质量等级"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def preprocess_for_ocr(self, image: np.ndarray, 
                          target_size: Tuple[int, int] = (120, 32)) -> np.ndarray:
        """专门为OCR优化的预处理"""
        try:
            # 1. 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 2. 调整大小
            width, height = target_size
            resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # 3. 归一化
            normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
            
            # 4. 高斯模糊轻微去噪
            denoised = cv2.GaussianBlur(normalized, (3, 3), 0)
            
            # 5. 二值化
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 6. 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return morphed
            
        except Exception as e:
            self.logger.warning(f"OCR预处理失败: {e}")
            return image
