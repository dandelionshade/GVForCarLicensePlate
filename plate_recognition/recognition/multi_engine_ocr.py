# -*- coding: utf-8 -*-
"""
OCR识别模块 - 多引擎OCR系统

将原来的multi_engine_ocr.py重构为更加模块化的识别系统
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.config import get_config
from ..core.exceptions import RecognitionError, ErrorCodes
from ..core.constants import PlateConstants, OCREngine


class BaseOCR(ABC):
    """OCR引擎基类"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.engine_name = "base"
    
    @abstractmethod
    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """识别文字
        
        Args:
            image: 输入图像
            
        Returns:
            识别结果字典：
            - text: 识别的文字
            - confidence: 置信度
            - engine: 引擎名称
            - processing_time: 处理时间
        """
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 调整大小
            height, width = gray.shape
            if height < 32:
                # 放大图像
                scale = 32 / height
                new_width = int(width * scale)
                gray = cv2.resize(gray, (new_width, 32), interpolation=cv2.INTER_CUBIC)
            
            # 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 去噪
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            return cleaned
            
        except Exception as e:
            self.logger.warning(f"图像预处理失败: {e}")
            return image
    
    def validate_result(self, text: str) -> Dict[str, Any]:
        """验证识别结果"""
        validation = {
            'is_valid': False,
            'confidence_penalty': 0.0,
            'issues': []
        }
        
        if not text:
            validation['issues'].append('识别结果为空')
            validation['confidence_penalty'] = 1.0
            return validation
        
        # 长度检查
        if len(text) < 7 or len(text) > 8:
            validation['issues'].append(f'长度异常: {len(text)}')
            validation['confidence_penalty'] += 0.3
        
        # 格式检查
        if not PlateConstants.validate_plate_number(text):
            validation['issues'].append('格式不符合车牌规范')
            validation['confidence_penalty'] += 0.4
        
        # 省份检查
        if text and text[0] not in PlateConstants.PROVINCES:
            validation['issues'].append(f'省份代码无效: {text[0]}')
            validation['confidence_penalty'] += 0.2
        
        # 字符检查
        for i, char in enumerate(text[1:], 1):
            if i == 1:  # 第二位应该是字母
                if char not in PlateConstants.LETTERS:
                    validation['issues'].append(f'第{i+1}位字符无效: {char}')
                    validation['confidence_penalty'] += 0.1
            else:  # 其他位应该是字母或数字
                if char not in PlateConstants.LETTERS + PlateConstants.DIGITS:
                    validation['issues'].append(f'第{i+1}位字符无效: {char}')
                    validation['confidence_penalty'] += 0.1
        
        # 限制惩罚最大值
        validation['confidence_penalty'] = min(validation['confidence_penalty'], 1.0)
        validation['is_valid'] = validation['confidence_penalty'] < 0.5
        
        return validation


class TesseractOCR(BaseOCR):
    """Tesseract OCR引擎"""
    
    def __init__(self):
        super().__init__()
        self.engine_name = OCREngine.TESSERACT.value
        self._initialize_tesseract()
    
    def _initialize_tesseract(self):
        """初始化Tesseract"""
        try:
            import pytesseract
            
            # 设置Tesseract路径
            if self.config.ocr.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.config.ocr.tesseract_cmd
            
            self.pytesseract = pytesseract
            self.available = True
            
        except ImportError:
            self.logger.error("pytesseract未安装")
            self.available = False
        except Exception as e:
            self.logger.error(f"Tesseract初始化失败: {e}")
            self.available = False
    
    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """使用Tesseract识别"""
        start_time = time.time()
        
        result = {
            'text': '',
            'confidence': 0.0,
            'engine': self.engine_name,
            'processing_time': 0.0,
            'available': self.available
        }
        
        if not self.available:
            result['error'] = 'Tesseract不可用'
            return result
        
        try:
            # 预处理图像
            processed_image = self.preprocess_image(image)
            
            # OCR识别
            config = self.config.ocr.tesseract_config
            
            # 获取详细结果
            data = self.pytesseract.image_to_data(
                processed_image, 
                config=config, 
                output_type=self.pytesseract.Output.DICT
            )
            
            # 提取文字和置信度
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            texts = [text for text, conf in zip(data['text'], data['conf']) 
                    if text.strip() and int(conf) > 0]
            
            if texts:
                # 拼接文字
                full_text = ''.join(texts).strip()
                
                # 计算平均置信度
                avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
                
                # 验证结果
                validation = self.validate_result(full_text)
                final_confidence = max(0.0, avg_confidence - validation['confidence_penalty'])
                
                result.update({
                    'text': full_text,
                    'confidence': final_confidence,
                    'raw_confidence': avg_confidence,
                    'validation': validation
                })
            
        except Exception as e:
            self.logger.error(f"Tesseract识别失败: {e}")
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result


class PaddleOCR(BaseOCR):
    """PaddleOCR引擎"""
    
    def __init__(self):
        super().__init__()
        self.engine_name = OCREngine.PADDLE.value
        self._initialize_paddle()
    
    def _initialize_paddle(self):
        """初始化PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=self.config.ocr.paddle_use_angle_cls,
                lang=self.config.ocr.paddle_lang,
                use_gpu=self.config.ocr.paddle_use_gpu,
                show_log=False
            )
            self.available = True
            
        except ImportError:
            self.logger.error("PaddleOCR未安装")
            self.available = False
        except Exception as e:
            self.logger.error(f"PaddleOCR初始化失败: {e}")
            self.available = False
    
    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """使用PaddleOCR识别"""
        start_time = time.time()
        
        result = {
            'text': '',
            'confidence': 0.0,
            'engine': self.engine_name,
            'processing_time': 0.0,
            'available': self.available
        }
        
        if not self.available:
            result['error'] = 'PaddleOCR不可用'
            return result
        
        try:
            # OCR识别
            ocr_result = self.paddle_ocr.ocr(image, cls=True)
            
            if ocr_result and ocr_result[0]:
                texts = []
                confidences = []
                
                for line in ocr_result[0]:
                    if len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        texts.append(text)
                        confidences.append(confidence)
                
                if texts:
                    # 拼接文字
                    full_text = ''.join(texts).strip()
                    
                    # 计算平均置信度
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    # 验证结果
                    validation = self.validate_result(full_text)
                    final_confidence = max(0.0, avg_confidence - validation['confidence_penalty'])
                    
                    result.update({
                        'text': full_text,
                        'confidence': final_confidence,
                        'raw_confidence': avg_confidence,
                        'validation': validation
                    })
            
        except Exception as e:
            self.logger.error(f"PaddleOCR识别失败: {e}")
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result


class GeminiOCR(BaseOCR):
    """Gemini Vision API OCR引擎"""
    
    def __init__(self):
        super().__init__()
        self.engine_name = OCREngine.GEMINI.value
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """初始化Gemini API"""
        try:
            import google.generativeai as genai
            
            if not self.config.gemini_api_key:
                self.logger.error("Gemini API密钥未配置")
                self.available = False
                return
            
            genai.configure(api_key=self.config.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-pro-vision')
            self.available = True
            
        except ImportError:
            self.logger.error("google-generativeai未安装")
            self.available = False
        except Exception as e:
            self.logger.error(f"Gemini初始化失败: {e}")
            self.available = False
    
    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """使用Gemini Vision API识别"""
        start_time = time.time()
        
        result = {
            'text': '',
            'confidence': 0.0,
            'engine': self.engine_name,
            'processing_time': 0.0,
            'available': self.available
        }
        
        if not self.available:
            result['error'] = 'Gemini API不可用'
            return result
        
        try:
            # 转换图像格式
            from PIL import Image
            
            # 转换为PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # 构造提示词
            prompt = """
            这是一张中国车牌图片。请识别出车牌号码，只返回车牌号码文字，不要包含任何其他内容。
            车牌格式通常是：省份简称+字母+5-6位数字字母组合
            例如：京A12345、沪B67890等
            请直接返回识别的车牌号码：
            """
            
            # 调用API
            response = self.model.generate_content([prompt, pil_image])
            
            if response and response.text:
                # 清理结果
                raw_text = response.text.strip()
                
                # 提取车牌号码（去除多余文字）
                import re
                plate_match = re.search(r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][0-9A-Z]{4,6}', raw_text)
                
                if plate_match:
                    plate_text = plate_match.group()
                    
                    # 验证结果
                    validation = self.validate_result(plate_text)
                    
                    # Gemini通常有较高的置信度
                    base_confidence = 0.85
                    final_confidence = max(0.0, base_confidence - validation['confidence_penalty'])
                    
                    result.update({
                        'text': plate_text,
                        'confidence': final_confidence,
                        'raw_confidence': base_confidence,
                        'raw_text': raw_text,
                        'validation': validation
                    })
                else:
                    result['error'] = f'未能从响应中提取车牌号码: {raw_text}'
            else:
                result['error'] = 'Gemini API无响应'
            
        except Exception as e:
            self.logger.error(f"Gemini识别失败: {e}")
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result


class MultiEngineOCR:
    """多引擎OCR融合器"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个引擎
        self.engines = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        """初始化所有OCR引擎"""
        # Tesseract
        tesseract = TesseractOCR()
        if tesseract.available:
            self.engines[OCREngine.TESSERACT.value] = tesseract
        
        # PaddleOCR
        paddle = PaddleOCR()
        if paddle.available:
            self.engines[OCREngine.PADDLE.value] = paddle
        
        # Gemini
        gemini = GeminiOCR()
        if gemini.available:
            self.engines[OCREngine.GEMINI.value] = gemini
        
        self.logger.info(f"可用OCR引擎: {list(self.engines.keys())}")
    
    def recognize(self, image: np.ndarray, 
                 engines: Optional[List[str]] = None,
                 parallel: bool = True) -> Dict[str, Any]:
        """多引擎识别
        
        Args:
            image: 输入图像
            engines: 指定使用的引擎列表
            parallel: 是否并行执行
            
        Returns:
            融合后的识别结果
        """
        start_time = time.time()
        
        # 确定使用的引擎
        target_engines = engines or list(self.engines.keys())
        available_engines = [e for e in target_engines if e in self.engines]
        
        if not available_engines:
            return {
                'success': False,
                'error': '没有可用的OCR引擎',
                'engines_used': [],
                'processing_time': time.time() - start_time
            }
        
        # 执行识别
        if parallel and len(available_engines) > 1:
            all_results = self._parallel_recognize(image, available_engines)
        else:
            all_results = self._sequential_recognize(image, available_engines)
        
        # 融合结果
        final_result = self._fuse_results(all_results)
        final_result['processing_time'] = time.time() - start_time
        final_result['engines_used'] = available_engines
        
        return final_result
    
    def _parallel_recognize(self, image: np.ndarray, 
                           engines: List[str]) -> Dict[str, Dict[str, Any]]:
        """并行识别"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(engines)) as executor:
            # 提交任务
            futures = {
                executor.submit(self.engines[engine].recognize, image): engine
                for engine in engines
            }
            
            # 收集结果
            for future in as_completed(futures):
                engine = futures[future]
                try:
                    result = future.result(timeout=30)  # 30秒超时
                    results[engine] = result
                except Exception as e:
                    self.logger.error(f"引擎 {engine} 识别失败: {e}")
                    results[engine] = {
                        'text': '',
                        'confidence': 0.0,
                        'engine': engine,
                        'error': str(e),
                        'available': False
                    }
        
        return results
    
    def _sequential_recognize(self, image: np.ndarray, 
                            engines: List[str]) -> Dict[str, Dict[str, Any]]:
        """顺序识别"""
        results = {}
        
        for engine in engines:
            try:
                result = self.engines[engine].recognize(image)
                results[engine] = result
            except Exception as e:
                self.logger.error(f"引擎 {engine} 识别失败: {e}")
                results[engine] = {
                    'text': '',
                    'confidence': 0.0,
                    'engine': engine,
                    'error': str(e),
                    'available': False
                }
        
        return results
    
    def _fuse_results(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """融合多引擎结果"""
        valid_results = []
        
        # 过滤有效结果
        for engine, result in all_results.items():
            if result.get('text') and result.get('confidence', 0) > 0:
                valid_results.append(result)
        
        if not valid_results:
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'best_engine': '',
                'all_results': all_results,
                'consensus': False
            }
        
        # 按置信度排序
        valid_results.sort(key=lambda x: x['confidence'], reverse=True)
        best_result = valid_results[0]
        
        # 检查一致性
        consensus = self._check_consensus(valid_results)
        
        # 如果有一致性，提升置信度
        final_confidence = best_result['confidence']
        if consensus['has_consensus']:
            final_confidence = min(1.0, final_confidence * 1.1)
        
        return {
            'success': True,
            'text': best_result['text'],
            'confidence': final_confidence,
            'best_engine': best_result['engine'],
            'all_results': all_results,
            'consensus': consensus['has_consensus'],
            'consensus_text': consensus.get('consensus_text', ''),
            'consensus_count': consensus.get('consensus_count', 0)
        }
    
    def _check_consensus(self, results: List[Dict[str, Any]], 
                        threshold: float = 0.6) -> Dict[str, Any]:
        """检查结果一致性"""
        if len(results) < 2:
            return {'has_consensus': False}
        
        # 统计相同结果
        text_counts = {}
        for result in results:
            text = result['text']
            if text:
                text_counts[text] = text_counts.get(text, 0) + 1
        
        if not text_counts:
            return {'has_consensus': False}
        
        # 找到出现次数最多的结果
        max_count = max(text_counts.values())
        consensus_texts = [text for text, count in text_counts.items() if count == max_count]
        
        # 检查是否达到阈值
        consensus_ratio = max_count / len(results)
        has_consensus = consensus_ratio >= threshold
        
        return {
            'has_consensus': has_consensus,
            'consensus_text': consensus_texts[0] if consensus_texts else '',
            'consensus_count': max_count,
            'consensus_ratio': consensus_ratio,
            'all_texts': text_counts
        }
    
    def get_available_engines(self) -> List[str]:
        """获取可用引擎列表"""
        return list(self.engines.keys())
    
    def is_engine_available(self, engine: str) -> bool:
        """检查引擎是否可用"""
        return engine in self.engines and self.engines[engine].available
