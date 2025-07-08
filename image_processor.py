# -*- coding: utf-8 -*-
"""
高级图像预处理模块 - 车牌识别系统
包含多种图像预处理和增强技术，用于提高识别准确率
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from skimage import exposure, restoration, filters, morphology
from skimage.util import random_noise
import imutils
from config import Config

class AdvancedImageProcessor:
    """高级图像处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def preprocess_image(self, image: np.ndarray, method: str = "comprehensive") -> Dict[str, np.ndarray]:
        """
        综合图像预处理
        
        Args:
            image: 输入图像
            method: 预处理方法 ("basic", "comprehensive", "adaptive")
            
        Returns:
            Dict: 包含多种预处理结果的字典
        """
        try:
            results = {}
            
            # 原始图像
            results['original'] = image.copy()
            
            # 基础预处理
            if method in ["basic", "comprehensive", "adaptive"]:
                results.update(self._basic_preprocessing(image))
            
            # 综合预处理
            if method in ["comprehensive", "adaptive"]:
                results.update(self._comprehensive_preprocessing(image))
            
            # 自适应预处理
            if method == "adaptive":
                results.update(self._adaptive_preprocessing(image))
                
            return results
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            return {"original": image}
    
    def _basic_preprocessing(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """基础预处理"""
        results = {}
        
        # 灰度化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        results['gray'] = gray
        
        # 高斯滤波去噪
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        results['denoised'] = denoised
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results['enhanced'] = enhanced
        
        # 二值化
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results['binary_otsu'] = binary_otsu
        
        # 自适应二值化
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        results['binary_adaptive'] = binary_adaptive
        
        return results
    
    def _comprehensive_preprocessing(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """综合预处理"""
        results = {}
        
        # 获取基础处理结果
        basic_results = self._basic_preprocessing(image)
        gray = basic_results['gray']
        
        # Retinex增强（解决光照不均）
        retinex = self._multiscale_retinex(gray)
        results['retinex'] = retinex
        
        # 顶帽变换（突出明亮区域）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        results['tophat'] = tophat
        
        # 黑帽变换（突出暗色区域）
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        results['blackhat'] = blackhat
        
        # 梯度增强
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient = np.uint8(gradient / gradient.max() * 255)
        results['gradient'] = gradient
        
        # 边缘增强
        edges_canny = cv2.Canny(gray, 50, 150)
        results['edges_canny'] = edges_canny
        
        # 形态学处理
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_close = cv2.morphologyEx(basic_results['binary_otsu'], cv2.MORPH_CLOSE, kernel_morph)
        morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel_morph)
        results['morph_processed'] = morph_open
        
        return results
    
    def _adaptive_preprocessing(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """自适应预处理（根据图像特征选择最佳参数）"""
        results = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 分析图像特征
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # 根据亮度和对比度自适应调整
        if brightness < 50:  # 低亮度图像
            # 增强亮度
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
            results['adaptive_brightness'] = enhanced
        elif brightness > 200:  # 高亮度图像
            # 降低亮度
            enhanced = cv2.convertScaleAbs(gray, alpha=0.7, beta=-20)
            results['adaptive_brightness'] = enhanced
        else:
            results['adaptive_brightness'] = gray
        
        if contrast < 30:  # 低对比度图像
            # 增强对比度
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            results['adaptive_contrast'] = enhanced
        else:
            results['adaptive_contrast'] = gray
            
        # 自适应去噪
        if np.std(gray) > 50:  # 高噪声图像
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            results['adaptive_denoise'] = denoised
        else:
            denoised = cv2.GaussianBlur(gray, (3, 3), 0)
            results['adaptive_denoise'] = denoised
        
        return results
    
    def _multiscale_retinex(self, image: np.ndarray, scales: List[int] = [15, 80, 250]) -> np.ndarray:
        """多尺度Retinex算法"""
        def single_scale_retinex(img, scale):
            blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), scale)
            retinex = np.log10(img.astype(np.float32) + 1) - np.log10(blurred + 1)
            return retinex
        
        retinex = np.zeros_like(image, dtype=np.float32)
        for scale in scales:
            retinex += single_scale_retinex(image, scale)
        
        retinex = retinex / len(scales)
        
        # 归一化到0-255
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255
        return retinex.astype(np.uint8)
    
    def enhance_plate_region(self, plate_image: np.ndarray) -> np.ndarray:
        """专门针对车牌区域的增强处理"""
        try:
            if plate_image is None or plate_image.size == 0:
                return plate_image
            
            # 确保是灰度图像
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image.copy()
            
            # 尺寸标准化
            height, width = gray.shape
            if height < 32 or width < 120:
                # 放大小尺寸图像
                scale_factor = max(32 / height, 120 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 车牌特定预处理
            # 1. 去除边框
            h, w = gray.shape
            roi = gray[int(h*0.1):int(h*0.9), int(w*0.05):int(w*0.95)]
            
            # 2. 字符分离增强
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            enhanced = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
            
            # 3. 对比度增强
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(enhanced)
            
            # 4. 锐化
            kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel_sharp)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"车牌区域增强失败: {e}")
            return plate_image
    
    def generate_augmented_data(self, image: np.ndarray, num_augments: int = 5) -> List[np.ndarray]:
        """生成数据增强样本"""
        augmented_images = []
        
        try:
            config = Config.AUGMENTATION_CONFIG
            
            for i in range(num_augments):
                augmented = image.copy()
                
                # 随机旋转
                if np.random.random() < 0.5:
                    angle = np.random.uniform(-config['rotation_range'], config['rotation_range'])
                    augmented = imutils.rotate_bound(augmented, angle)
                
                # 随机平移
                if np.random.random() < 0.5:
                    shift_x = int(np.random.uniform(-config['width_shift_range'], config['width_shift_range']) * image.shape[1])
                    shift_y = int(np.random.uniform(-config['height_shift_range'], config['height_shift_range']) * image.shape[0])
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    augmented = cv2.warpAffine(augmented, M, (image.shape[1], image.shape[0]))
                
                # 随机亮度调整
                if np.random.random() < 0.5:
                    brightness = np.random.uniform(config['brightness_range'][0], config['brightness_range'][1])
                    augmented = cv2.convertScaleAbs(augmented, alpha=brightness, beta=0)
                
                # 添加噪声
                if np.random.random() < 0.3:
                    noise = random_noise(augmented / 255.0, mode='gaussian', var=config['noise_factor'])
                    augmented = (noise * 255).astype(np.uint8)
                
                augmented_images.append(augmented)
            
            return augmented_images
            
        except Exception as e:
            self.logger.error(f"数据增强失败: {e}")
            return [image]
    
    def remove_background_noise(self, image: np.ndarray) -> np.ndarray:
        """去除背景噪声"""
        try:
            # 形态学开运算去除小噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
            # 中值滤波去除椒盐噪声
            cleaned = cv2.medianBlur(cleaned, 3)
            
            # 双边滤波保持边缘同时去噪
            cleaned = cv2.bilateralFilter(cleaned, 9, 75, 75)
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"背景噪声去除失败: {e}")
            return image
    
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """倾斜校正"""
        try:
            # 检测直线
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # 计算主要角度
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi - 90
                    if abs(angle) < 45:  # 只考虑小角度倾斜
                        angles.append(angle)
                
                if angles:
                    # 使用中位数角度进行校正
                    correction_angle = np.median(angles)
                    
                    # 旋转图像
                    corrected = imutils.rotate_bound(image, correction_angle)
                    return corrected
            
            return image
            
        except Exception as e:
            self.logger.error(f"倾斜校正失败: {e}")
            return image
