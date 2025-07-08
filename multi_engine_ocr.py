# -*- coding: utf-8 -*-
"""
多引擎OCR识别模块 - 车牌识别系统
整合Tesseract、PaddleOCR和Gemini等多种OCR引擎
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config, ErrorCodes

# 尝试导入各种OCR库
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract不可用")

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logging.warning("PaddleOCR不可用")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Gemini不可用")

class MultiEngineOCR:
    """多引擎OCR识别器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self._init_engines()
        
    def _init_engines(self):
        """初始化OCR引擎"""
        self.engines = {}
        
        # 初始化Tesseract
        if TESSERACT_AVAILABLE:
            try:
                pytesseract.pytesseract.tesseract_cmd = self.config.TESSERACT_CMD
                self.engines['tesseract'] = True
                self.logger.info("Tesseract引擎初始化成功")
            except Exception as e:
                self.logger.error(f"Tesseract初始化失败: {e}")
                self.engines['tesseract'] = False
        
        # 初始化PaddleOCR
        if PADDLE_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=self.config.PADDLE_USE_ANGLE_CLS,
                    lang=self.config.PADDLE_LANG,
                    use_gpu=self.config.PADDLE_USE_GPU,
                    show_log=False
                )
                self.engines['paddle'] = True
                self.logger.info("PaddleOCR引擎初始化成功")
            except Exception as e:
                self.logger.error(f"PaddleOCR初始化失败: {e}")
                self.engines['paddle'] = False
        
        # 初始化Gemini
        if GEMINI_AVAILABLE and self.config.GEMINI_API_KEY:
            try:
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.engines['gemini'] = True
                self.logger.info("Gemini引擎初始化成功")
            except Exception as e:
                self.logger.error(f"Gemini初始化失败: {e}")
                self.engines['gemini'] = False
    
    def recognize(self, image: np.ndarray, engines: List[str] = None, parallel: bool = True) -> Dict[str, Any]:
        """
        多引擎车牌识别
        
        Args:
            image: 输入图像
            engines: 使用的引擎列表，None表示使用所有可用引擎
            parallel: 是否并行执行
            
        Returns:
            Dict: 识别结果
        """
        if engines is None:
            engines = [name for name, available in self.engines.items() if available]
        
        if not engines:
            return {
                'success': False,
                'error_code': ErrorCodes.OCR_ERROR,
                'message': '没有可用的OCR引擎'
            }
        
        start_time = time.time()
        
        if parallel and len(engines) > 1:
            results = self._parallel_recognize(image, engines)
        else:
            results = self._sequential_recognize(image, engines)
        
        # 融合结果
        final_result = self._fuse_results(results)
        final_result['processing_time'] = time.time() - start_time
        final_result['engines_used'] = engines
        
        return final_result
    
    def _parallel_recognize(self, image: np.ndarray, engines: List[str]) -> Dict[str, Dict[str, Any]]:
        """并行识别"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(engines)) as executor:
            # 提交任务
            future_to_engine = {}
            for engine in engines:
                if engine == 'tesseract':
                    future = executor.submit(self._recognize_tesseract, image)
                elif engine == 'paddle':
                    future = executor.submit(self._recognize_paddle, image)
                elif engine == 'gemini':
                    future = executor.submit(self._recognize_gemini, image)
                else:
                    continue
                
                future_to_engine[future] = engine
            
            # 收集结果
            for future in as_completed(future_to_engine, timeout=self.config.MAX_REQUEST_TIME):
                engine = future_to_engine[future]
                try:
                    result = future.result()
                    results[engine] = result
                except Exception as e:
                    self.logger.error(f"{engine}识别失败: {e}")
                    results[engine] = {
                        'success': False,
                        'text': '',
                        'confidence': 0.0,
                        'error': str(e)
                    }
        
        return results
    
    def _sequential_recognize(self, image: np.ndarray, engines: List[str]) -> Dict[str, Dict[str, Any]]:
        """顺序识别"""
        results = {}
        
        for engine in engines:
            try:
                if engine == 'tesseract':
                    results[engine] = self._recognize_tesseract(image)
                elif engine == 'paddle':
                    results[engine] = self._recognize_paddle(image)
                elif engine == 'gemini':
                    results[engine] = self._recognize_gemini(image)
            except Exception as e:
                self.logger.error(f"{engine}识别失败: {e}")
                results[engine] = {
                    'success': False,
                    'text': '',
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return results
    
    def _recognize_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """Tesseract OCR识别"""
        if not self.engines.get('tesseract', False):
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': 'Tesseract不可用'
            }
        
        try:
            start_time = time.time()
            
            # 预处理图像
            processed_images = self._preprocess_for_tesseract(image)
            
            best_result = None
            best_score = 0
            
            for method_name, processed_img in processed_images.items():
                try:
                    # OCR识别
                    text = pytesseract.image_to_string(
                        processed_img,
                        config=self.config.TESSERACT_CONFIG,
                        lang='chi_sim+eng'
                    )
                    
                    # 清理文本
                    cleaned_text = self._clean_plate_text(text)
                    
                    if cleaned_text:
                        # 计算置信度
                        confidence = self._calculate_text_confidence(cleaned_text, 'tesseract')
                        score = confidence * len(cleaned_text)
                        
                        if score > best_score:
                            best_score = score
                            best_result = {
                                'text': cleaned_text,
                                'confidence': confidence,
                                'method': method_name
                            }
                
                except Exception as e:
                    self.logger.debug(f"Tesseract方法{method_name}失败: {e}")
                    continue
            
            if best_result:
                return {
                    'success': True,
                    'text': best_result['text'],
                    'confidence': best_result['confidence'],
                    'method': best_result['method'],
                    'processing_time': time.time() - start_time
                }
            else:
                return {
                    'success': False,
                    'text': '',
                    'confidence': 0.0,
                    'error': '未识别到有效文本'
                }
                
        except Exception as e:
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _recognize_paddle(self, image: np.ndarray) -> Dict[str, Any]:
        """PaddleOCR识别"""
        if not self.engines.get('paddle', False):
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': 'PaddleOCR不可用'
            }
        
        try:
            start_time = time.time()
            
            # PaddleOCR识别
            results = self.paddle_ocr.ocr(image, cls=True)
            
            if not results or not results[0]:
                return {
                    'success': False,
                    'text': '',
                    'confidence': 0.0,
                    'error': '未检测到文本'
                }
            
            # 提取最佳结果
            best_text = ''
            best_confidence = 0.0
            
            for line in results[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], tuple) else line[1]
                    confidence = line[1][1] if isinstance(line[1], tuple) else 0.8
                    
                    # 清理文本
                    cleaned_text = self._clean_plate_text(text)
                    
                    if cleaned_text and confidence > best_confidence:
                        best_text = cleaned_text
                        best_confidence = confidence
            
            if best_text:
                return {
                    'success': True,
                    'text': best_text,
                    'confidence': best_confidence,
                    'processing_time': time.time() - start_time
                }
            else:
                return {
                    'success': False,
                    'text': '',
                    'confidence': 0.0,
                    'error': '未识别到有效车牌号'
                }
                
        except Exception as e:
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _recognize_gemini(self, image: np.ndarray) -> Dict[str, Any]:
        """Gemini Vision识别"""
        if not self.engines.get('gemini', False):
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': 'Gemini不可用'
            }
        
        try:
            import base64
            import json
            
            start_time = time.time()
            
            # 编码图像
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 构建提示词
            prompt = """
            请识别这张中国车牌图片中的车牌号码。
            
            要求：
            1. 只返回车牌号码，不要其他内容
            2. 中国车牌格式：省份简称 + 字母 + 数字字母组合
            3. 如果无法识别，返回"UNKNOWN"
            
            请严格按照以下JSON格式返回：
            {
                "plate_number": "识别的车牌号码",
                "confidence": 0.95
            }
            """
            
            # 调用API
            response = self.gemini_model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_base64}
            ])
            
            if not response.text:
                return {
                    'success': False,
                    'text': '',
                    'confidence': 0.0,
                    'error': 'API返回空响应'
                }
            
            # 解析响应
            cleaned_response = response.text.strip()
            cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
            
            try:
                result_data = json.loads(cleaned_response)
                plate_number = result_data.get('plate_number', '')
                confidence = result_data.get('confidence', 0.5)
                
                # 清理文本
                cleaned_text = self._clean_plate_text(plate_number)
                
                if cleaned_text and cleaned_text != "UNKNOWN":
                    return {
                        'success': True,
                        'text': cleaned_text,
                        'confidence': float(confidence),
                        'processing_time': time.time() - start_time
                    }
                else:
                    return {
                        'success': False,
                        'text': '',
                        'confidence': 0.0,
                        'error': '未识别到有效车牌号'
                    }
                    
            except json.JSONDecodeError:
                # 如果JSON解析失败，尝试直接提取车牌号
                cleaned_text = self._clean_plate_text(cleaned_response)
                if cleaned_text and len(cleaned_text) >= 6:
                    return {
                        'success': True,
                        'text': cleaned_text,
                        'confidence': 0.7,
                        'processing_time': time.time() - start_time
                    }
                else:
                    return {
                        'success': False,
                        'text': '',
                        'confidence': 0.0,
                        'error': '响应格式错误'
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _preprocess_for_tesseract(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """为Tesseract预处理图像"""
        processed = {}
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 方法1：原始灰度图
        processed['original'] = gray
        
        # 方法2：降噪
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        processed['denoised'] = denoised
        
        # 方法3：OTSU二值化
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed['binary_otsu'] = binary_otsu
        
        # 方法4：自适应二值化
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        processed['binary_adaptive'] = binary_adaptive
        
        # 方法5：形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel)
        processed['morphology'] = morph
        
        # 方法6：对比度增强
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed['enhanced'] = enhanced
        
        return processed
    
    def _clean_plate_text(self, text: str) -> str:
        """清理车牌文本"""
        if not text:
            return ""
        
        # 移除空白字符
        cleaned = re.sub(r'\s+', '', text)
        
        # 只保留中文、英文字母和数字
        cleaned = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]', '', cleaned)
        
        # 转换为大写
        cleaned = cleaned.upper()
        
        # 常见字符纠正
        corrections = {
            '0': 'O',  # 在某些位置0应该是O
            'I': '1',  # I容易误识别为1
            'Z': '2',  # Z容易误识别为2
            'S': '5',  # S容易误识别为5
            'G': '6',  # G容易误识别为6
            'B': '8',  # B容易误识别为8
        }
        
        # 根据位置进行纠正
        if len(cleaned) >= 2:
            # 第二位应该是字母
            if cleaned[1].isdigit():
                for digit, letter in corrections.items():
                    if cleaned[1] == digit:
                        cleaned = cleaned[0] + letter + cleaned[2:]
                        break
        
        return cleaned
    
    def _calculate_text_confidence(self, text: str, engine: str) -> float:
        """计算文本置信度"""
        if not text:
            return 0.0
        
        confidence = 0.0
        
        # 长度检查
        if 6 <= len(text) <= 8:
            confidence += 0.3
        elif 5 <= len(text) <= 9:
            confidence += 0.1
        
        # 格式检查
        plate_pattern = self.config.PLATE_PATTERN
        if re.match(plate_pattern, text):
            confidence += 0.4
        
        # 字符类型检查
        has_chinese = any('\u4e00' <= c <= '\u9fa5' for c in text)
        has_letter = any(c.isalpha() and c.isascii() for c in text)
        has_digit = any(c.isdigit() for c in text)
        
        if has_chinese:
            confidence += 0.2
        if has_letter:
            confidence += 0.1
        if has_digit:
            confidence += 0.1
        
        # 引擎特定调整
        if engine == 'paddle':
            confidence *= 1.1  # PaddleOCR对中文识别较好
        elif engine == 'gemini':
            confidence *= 1.05  # Gemini有较好的上下文理解
        
        return min(confidence, 1.0)
    
    def _fuse_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """融合多引擎识别结果"""
        if not results:
            return {
                'success': False,
                'error_code': ErrorCodes.OCR_ERROR,
                'message': '没有识别结果'
            }
        
        # 过滤成功的结果
        successful_results = {
            engine: result for engine, result in results.items()
            if result.get('success', False) and result.get('text', '')
        }
        
        if not successful_results:
            return {
                'success': False,
                'error_code': ErrorCodes.OCR_ERROR,
                'message': '所有引擎都识别失败',
                'details': results
            }
        
        # 按置信度排序
        sorted_results = sorted(
            successful_results.items(),
            key=lambda x: x[1].get('confidence', 0),
            reverse=True
        )
        
        # 选择最佳结果
        best_engine, best_result = sorted_results[0]
        
        # 检查是否有多个引擎给出相同结果
        text_votes = {}
        for engine, result in successful_results.items():
            text = result.get('text', '')
            if text:
                if text not in text_votes:
                    text_votes[text] = []
                text_votes[text].append((engine, result.get('confidence', 0)))
        
        # 如果有多个引擎给出相同结果，提升置信度
        final_text = best_result.get('text', '')
        final_confidence = best_result.get('confidence', 0)
        
        if final_text in text_votes and len(text_votes[final_text]) > 1:
            # 多引擎一致，提升置信度
            avg_confidence = sum(conf for _, conf in text_votes[final_text]) / len(text_votes[final_text])
            final_confidence = min(avg_confidence * 1.2, 1.0)
        
        return {
            'success': True,
            'text': final_text,
            'confidence': final_confidence,
            'best_engine': best_engine,
            'all_results': results,
            'consensus': len(text_votes.get(final_text, [])) > 1
        }
