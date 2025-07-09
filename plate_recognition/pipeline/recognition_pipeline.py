# -*- coding: utf-8 -*-
"""
车牌识别处理流水线

整合检测、识别、预处理等各个模块，提供统一的识别接口
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from ..core.config import get_config
from ..core.exceptions import PlateRecognitionError, ErrorCodes
from ..core.constants import PlateConstants
from ..detection.detector import PlateDetector
from ..recognition.multi_engine_ocr import MultiEngineOCR
from ..preprocessing.image_processor import ImageProcessor


class RecognitionPipeline:
    """车牌识别流水线主类"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """初始化流水线
        
        Args:
            config_override: 配置覆盖参数
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # 应用配置覆盖
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # 初始化各个组件
        self._initialize_components()
        
        # 性能统计
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0
        }
    
    def _initialize_components(self):
        """初始化流水线组件"""
        try:
            # 图像预处理器
            self.image_processor = ImageProcessor()
            self.logger.info("图像处理器初始化成功")
        except Exception as e:
            self.logger.error(f"图像处理器初始化失败: {e}")
            self.image_processor = None
        
        try:
            # 车牌检测器
            self.plate_detector = PlateDetector()
            self.logger.info("车牌检测器初始化成功")
        except Exception as e:
            self.logger.error(f"车牌检测器初始化失败: {e}")
            self.plate_detector = None
        
        try:
            # OCR识别器
            self.ocr_engine = MultiEngineOCR()
            available_engines = self.ocr_engine.get_available_engines()
            self.logger.info(f"OCR引擎初始化成功，可用引擎: {available_engines}")
        except Exception as e:
            self.logger.error(f"OCR引擎初始化失败: {e}")
            self.ocr_engine = None
    
    def recognize_from_file(self, image_path: Union[str, Path], 
                           **kwargs) -> Dict[str, Any]:
        """从文件识别车牌
        
        Args:
            image_path: 图像文件路径
            **kwargs: 其他参数
            
        Returns:
            识别结果字典
        """
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            
            if image is None:
                raise PlateRecognitionError(
                    f"无法读取图像文件: {image_path}", 
                    ErrorCodes.IMAGE_LOAD_ERROR
                )
            
            # 调用主识别方法
            result = self.recognize(image, **kwargs)
            result['source_file'] = str(image_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"从文件识别失败 {image_path}: {e}")
            if isinstance(e, PlateRecognitionError):
                raise
            else:
                raise PlateRecognitionError(
                    f"从文件识别失败: {e}", 
                    ErrorCodes.UNKNOWN_ERROR
                )
    
    def recognize(self, image: np.ndarray,
                 detection_method: str = "comprehensive",
                 ocr_engines: Optional[List[str]] = None,
                 enable_preprocessing: bool = True,
                 return_debug_info: bool = False) -> Dict[str, Any]:
        """主要识别方法
        
        Args:
            image: 输入图像
            detection_method: 检测方法
            ocr_engines: 指定OCR引擎
            enable_preprocessing: 是否启用预处理
            return_debug_info: 是否返回调试信息
            
        Returns:
            识别结果字典
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        # 初始化结果
        result = {
            'success': False,
            'plate_number': '',
            'confidence': 0.0,
            'processing_time': 0.0,
            'pipeline_stages': {},
            'error': None
        }
        
        try:
            # 验证输入
            if image is None or image.size == 0:
                raise PlateRecognitionError(
                    "输入图像无效", 
                    ErrorCodes.IMAGE_LOAD_ERROR
                )
            
            debug_info = {} if return_debug_info else None
            
            # 阶段1: 图像预处理
            processed_image = self._preprocess_stage(
                image, enable_preprocessing, debug_info
            )
            
            # 阶段2: 车牌检测
            detections = self._detection_stage(
                processed_image, detection_method, debug_info
            )
            
            # 阶段3: OCR识别
            recognition_result = self._recognition_stage(
                processed_image, detections, ocr_engines, debug_info
            )
            
            # 阶段4: 结果后处理
            final_result = self._postprocess_stage(
                recognition_result, debug_info
            )
            
            # 更新结果
            result.update(final_result)
            result['success'] = bool(result['plate_number'])
            
            if result['success']:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            
            if return_debug_info and debug_info:
                result['debug_info'] = debug_info
            
        except PlateRecognitionError as e:
            self.logger.error(f"识别失败: {e}")
            result['error'] = str(e)
            result['error_code'] = e.error_code
            self.stats['failed_requests'] += 1
            
        except Exception as e:
            self.logger.error(f"未知错误: {e}")
            result['error'] = f"处理失败: {e}"
            result['error_code'] = ErrorCodes.UNKNOWN_ERROR
            self.stats['failed_requests'] += 1
        
        # 更新处理时间统计
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        self._update_average_time(processing_time)
        
        return result
    
    def _preprocess_stage(self, image: np.ndarray, 
                         enable_preprocessing: bool,
                         debug_info: Optional[Dict]) -> np.ndarray:
        """预处理阶段"""
        stage_start = time.time()
        
        try:
            if enable_preprocessing and self.image_processor:
                # 使用高级预处理
                processed_result = self.image_processor.preprocess_image(
                    image, method="comprehensive"
                )
                
                # 选择最好的预处理结果
                if 'enhanced' in processed_result:
                    processed_image = processed_result['enhanced']
                elif 'denoised' in processed_result:
                    processed_image = processed_result['denoised']
                else:
                    processed_image = image
                
                if debug_info is not None:
                    debug_info['preprocessing'] = {
                        'methods_applied': list(processed_result.keys()),
                        'processing_time': time.time() - stage_start
                    }
            else:
                # 基础预处理
                processed_image = self._basic_preprocess(image)
                
                if debug_info is not None:
                    debug_info['preprocessing'] = {
                        'methods_applied': ['basic'],
                        'processing_time': time.time() - stage_start
                    }
            
            return processed_image
            
        except Exception as e:
            self.logger.warning(f"预处理失败，使用原图: {e}")
            return image
    
    def _basic_preprocess(self, image: np.ndarray) -> np.ndarray:
        """基础预处理"""
        try:
            # 调整图像大小
            height, width = image.shape[:2]
            max_width, max_height = self.config.image_processing.max_size
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"基础预处理失败: {e}")
            return image
    
    def _detection_stage(self, image: np.ndarray, 
                        detection_method: str,
                        debug_info: Optional[Dict]) -> List[Dict[str, Any]]:
        """检测阶段"""
        stage_start = time.time()
        
        if not self.plate_detector:
            raise PlateRecognitionError(
                "车牌检测器不可用", 
                ErrorCodes.DETECTION_ERROR
            )
        
        try:
            detections = self.plate_detector.detect_plates(image, detection_method)
            
            if debug_info is not None:
                debug_info['detection'] = {
                    'method': detection_method,
                    'candidates_found': len(detections),
                    'processing_time': time.time() - stage_start,
                    'candidates': [
                        {
                            'bbox': d['bbox'],
                            'confidence': d['confidence'],
                            'method': d.get('method', ''),
                            'area': d.get('area', 0),
                            'aspect_ratio': d.get('aspect_ratio', 0)
                        }
                        for d in detections[:3]  # 只保存前3个
                    ]
                }
            
            return detections
            
        except Exception as e:
            raise PlateRecognitionError(
                f"车牌检测失败: {e}", 
                ErrorCodes.DETECTION_ERROR
            )
    
    def _recognition_stage(self, image: np.ndarray,
                          detections: List[Dict[str, Any]],
                          ocr_engines: Optional[List[str]],
                          debug_info: Optional[Dict]) -> Dict[str, Any]:
        """识别阶段"""
        stage_start = time.time()
        
        if not self.ocr_engine:
            raise PlateRecognitionError(
                "OCR引擎不可用", 
                ErrorCodes.OCR_ERROR
            )
        
        try:
            recognition_results = []
            
            if detections:
                # 对每个检测结果进行OCR
                for i, detection in enumerate(detections[:3]):  # 最多处理前3个
                    region = detection.get('region')
                    if region is not None and region.size > 0:
                        ocr_result = self.ocr_engine.recognize(
                            region, engines=ocr_engines, parallel=True
                        )
                        
                        if ocr_result.get('success'):
                            # 结合检测和识别的置信度
                            detection_conf = detection.get('confidence', 0.0)
                            ocr_conf = ocr_result.get('confidence', 0.0)
                            combined_conf = (detection_conf * 0.3 + ocr_conf * 0.7)
                            
                            recognition_results.append({
                                'detection_index': i,
                                'text': ocr_result.get('text', ''),
                                'confidence': combined_conf,
                                'detection_confidence': detection_conf,
                                'ocr_confidence': ocr_conf,
                                'engine': ocr_result.get('best_engine', ''),
                                'bbox': detection['bbox']
                            })
            else:
                # 如果没有检测到车牌，对整图进行OCR
                ocr_result = self.ocr_engine.recognize(
                    image, engines=ocr_engines, parallel=True
                )
                
                if ocr_result.get('success'):
                    recognition_results.append({
                        'detection_index': -1,
                        'text': ocr_result.get('text', ''),
                        'confidence': ocr_result.get('confidence', 0.0),
                        'detection_confidence': 0.0,
                        'ocr_confidence': ocr_result.get('confidence', 0.0),
                        'engine': ocr_result.get('best_engine', ''),
                        'bbox': None
                    })
            
            if debug_info is not None:
                debug_info['recognition'] = {
                    'processing_time': time.time() - stage_start,
                    'regions_processed': len(recognition_results),
                    'results': recognition_results
                }
            
            return {'results': recognition_results}
            
        except Exception as e:
            raise PlateRecognitionError(
                f"OCR识别失败: {e}", 
                ErrorCodes.OCR_ERROR
            )
    
    def _postprocess_stage(self, recognition_result: Dict[str, Any],
                          debug_info: Optional[Dict]) -> Dict[str, Any]:
        """后处理阶段"""
        stage_start = time.time()
        
        results = recognition_result.get('results', [])
        
        if not results:
            return {
                'plate_number': '',
                'confidence': 0.0,
                'engine_used': '',
                'detection_info': {}
            }
        
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        best_result = results[0]
        
        # 应用后处理规则
        final_text = self._apply_postprocess_rules(best_result['text'])
        
        # 最终验证
        validation = self._final_validation(final_text)
        
        final_confidence = best_result['confidence']
        if not validation['is_valid']:
            final_confidence *= 0.5  # 降低置信度
        
        result = {
            'plate_number': final_text,
            'confidence': final_confidence,
            'engine_used': best_result['engine'],
            'detection_info': {
                'bbox': best_result['bbox'],
                'detection_confidence': best_result['detection_confidence'],
                'ocr_confidence': best_result['ocr_confidence']
            },
            'validation': validation,
            'all_results': results
        }
        
        if debug_info is not None:
            debug_info['postprocessing'] = {
                'processing_time': time.time() - stage_start,
                'rules_applied': True,
                'validation_result': validation
            }
        
        return result
    
    def _apply_postprocess_rules(self, text: str) -> str:
        """应用后处理规则"""
        if not text:
            return text
        
        # 去除空格和特殊字符
        cleaned = ''.join(c for c in text if c.isalnum() or c in PlateConstants.PROVINCES)
        
        # 常见OCR错误修正
        corrections = {
            '0': 'D',  # 在第二位时，0可能是D
            'I': '1',  # I通常是1
            'O': '0',  # O通常是0
        }
        
        # 应用修正规则
        result = list(cleaned)
        
        for i, char in enumerate(result):
            # 第二位必须是字母
            if i == 1 and char.isdigit():
                if char in corrections:
                    result[i] = corrections[char]
            
            # 其他位置的常见错误
            elif i > 1 and char in corrections:
                # 根据上下文决定是否修正
                pass
        
        return ''.join(result)
    
    def _final_validation(self, text: str) -> Dict[str, Any]:
        """最终验证"""
        validation = {
            'is_valid': False,
            'issues': [],
            'score': 0.0
        }
        
        if not text:
            validation['issues'].append('结果为空')
            return validation
        
        # 长度检查
        if not (7 <= len(text) <= 8):
            validation['issues'].append(f'长度异常: {len(text)}')
        
        # 格式检查
        if not PlateConstants.validate_plate_number(text):
            validation['issues'].append('格式不符合规范')
        
        # 计算得分
        score = 1.0 - (len(validation['issues']) * 0.2)
        validation['score'] = max(0.0, score)
        validation['is_valid'] = validation['score'] >= 0.6
        
        return validation
    
    def _update_average_time(self, processing_time: float):
        """更新平均处理时间"""
        total = self.stats['total_requests']
        current_avg = self.stats['average_processing_time']
        
        # 计算移动平均
        self.stats['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def batch_recognize(self, image_paths: List[Union[str, Path]],
                       **kwargs) -> List[Dict[str, Any]]:
        """批量识别
        
        Args:
            image_paths: 图像文件路径列表
            **kwargs: 识别参数
            
        Returns:
            识别结果列表
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"处理第 {i+1}/{len(image_paths)} 张图片: {image_path}")
            
            try:
                result = self.recognize_from_file(image_path, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"处理图片失败 {image_path}: {e}")
                results.append({
                    'success': False,
                    'source_file': str(image_path),
                    'error': str(e),
                    'plate_number': '',
                    'confidence': 0.0
                })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        total = self.stats['total_requests']
        success_rate = (
            self.stats['successful_requests'] / total * 100 
            if total > 0 else 0
        )
        
        return {
            'total_requests': total,
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': f"{success_rate:.2f}%",
            'average_processing_time': f"{self.stats['average_processing_time']:.3f}s",
            'component_status': {
                'image_processor': self.image_processor is not None,
                'plate_detector': self.plate_detector is not None,
                'ocr_engine': self.ocr_engine is not None,
                'available_ocr_engines': (
                    self.ocr_engine.get_available_engines() 
                    if self.ocr_engine else []
                )
            }
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0
        }
