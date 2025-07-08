"""
优化车牌识别系统 - 主集成模块
整合所有优化模块，提供统一的高级识别接口
"""
import cv2
import numpy as np
import logging
import threading
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入现有模块（使用try-except确保兼容性）
try:
    from config import Config, ErrorCodes
    from image_processor import AdvancedImageProcessor
    from plate_detector import MultiAlgorithmPlateDetector
    from multi_engine_ocr import MultiEngineOCR
    from crnn_model import CRNNModel
    from test_framework import PlateRecognitionTestSuite
    from api_server import FlaskAPIServer
except ImportError as e:
    print(f"模块导入警告: {e}")
    print("某些高级功能可能不可用，但基础功能仍可正常运行")

class OptimizedPlateRecognitionSystem:
    """优化的车牌识别系统"""
    
    def __init__(self, enable_gpu: bool = False, enable_caching: bool = True):
        """
        初始化优化的车牌识别系统
        
        Args:
            enable_gpu: 是否启用GPU加速
            enable_caching: 是否启用结果缓存
        """
        self.logger = self._setup_logging()
        self.config = Config()
        self.enable_gpu = enable_gpu
        self.enable_caching = enable_caching
        
        # 初始化各组件
        self._initialize_components()
        
        # 性能监控
        self.performance_metrics = {
            'total_processed': 0,
            'success_count': 0,
            'average_time': 0.0,
            'error_count': 0,
            'accuracy_rate': 0.0
        }
        
        # 结果缓存
        self.result_cache = {} if enable_caching else None
        
        self.logger.info("优化车牌识别系统初始化完成")
    
    def _setup_logging(self):
        """设置日志"""
        logger = logging.getLogger("OptimizedPlateRecognition")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_components(self):
        """初始化系统组件"""
        try:
            # 图像处理器
            self.image_processor = AdvancedImageProcessor()
            self.logger.info("图像处理器初始化成功")
        except:
            self.image_processor = None
            self.logger.warning("高级图像处理器不可用，使用基础处理")
        
        try:
            # 车牌检测器
            self.plate_detector = MultiAlgorithmPlateDetector()
            self.logger.info("车牌检测器初始化成功")
        except:
            self.plate_detector = None
            self.logger.warning("高级车牌检测器不可用")
        
        try:
            # 多引擎OCR
            self.ocr_engine = MultiEngineOCR()
            self.logger.info("多引擎OCR初始化成功")
        except:
            self.ocr_engine = None
            self.logger.warning("多引擎OCR不可用")
        
        try:
            # CRNN模型
            self.crnn_model = CRNNModel()
            if hasattr(self.crnn_model, 'load_model'):
                try:
                    self.crnn_model.load_model()
                    self.logger.info("CRNN模型加载成功")
                except:
                    self.logger.warning("CRNN模型加载失败，将跳过CRNN识别")
        except:
            self.crnn_model = None
            self.logger.warning("CRNN模型不可用")
    
    def recognize_plate(self, image: np.ndarray, method: str = "fusion") -> Dict[str, Any]:
        """
        车牌识别主接口
        
        Args:
            image: 输入图像
            method: 识别方法 ("basic", "advanced", "fusion")
            
        Returns:
            识别结果字典
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            if self.enable_caching:
                cache_key = self._generate_cache_key(image)
                if cache_key in self.result_cache:
                    self.logger.info("从缓存获取结果")
                    return self.result_cache[cache_key]
            
            # 基础识别
            if method == "basic":
                result = self._basic_recognition(image)
            # 高级识别
            elif method == "advanced":
                result = self._advanced_recognition(image)
            # 融合识别
            elif method == "fusion":
                result = self._fusion_recognition(image)
            else:
                raise ValueError(f"不支持的识别方法: {method}")
            
            # 计算耗时
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            # 更新性能指标
            self._update_performance_metrics(result, processing_time)
            
            # 缓存结果
            if self.enable_caching and cache_key:
                self.result_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"车牌识别失败: {e}")
            self.performance_metrics['error_count'] += 1
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _basic_recognition(self, image: np.ndarray) -> Dict[str, Any]:
        """基础识别流程"""
        result = {
            'success': False,
            'method': 'basic',
            'plate_text': '',
            'confidence': 0.0,
            'details': {}
        }
        
        try:
            # 简单的图像预处理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # 使用OpenCV进行基础车牌检测
            plate_regions = self._basic_plate_detection(gray)
            
            if not plate_regions:
                result['details']['error'] = '未检测到车牌区域'
                return result
            
            # 对最大区域进行OCR
            largest_region = max(plate_regions, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_region
            plate_img = gray[y:y+h, x:x+w]
            
            # 基础OCR识别
            plate_text = self._basic_ocr(plate_img)
            
            if plate_text:
                result['success'] = True
                result['plate_text'] = plate_text
                result['confidence'] = 0.7  # 基础方法给予中等置信度
                result['details']['detection_method'] = 'opencv_contours'
                result['details']['ocr_method'] = 'tesseract'
            
            return result
            
        except Exception as e:
            result['details']['error'] = str(e)
            return result
    
    def _advanced_recognition(self, image: np.ndarray) -> Dict[str, Any]:
        """高级识别流程"""
        result = {
            'success': False,
            'method': 'advanced',
            'plate_text': '',
            'confidence': 0.0,
            'details': {}
        }
        
        try:
            # 高级图像预处理
            if self.image_processor:
                processed_images = self.image_processor.preprocess_image(image, "comprehensive")
                best_image = processed_images.get('enhanced', image)
            else:
                best_image = image
            
            # 高级车牌检测
            if self.plate_detector:
                detections = self.plate_detector.detect_plates(best_image)
                if not detections:
                    result['details']['error'] = '高级检测器未找到车牌'
                    return result
                
                # 选择最佳检测结果
                best_detection = max(detections, key=lambda x: x.get('confidence', 0))
                plate_img = best_detection['plate_image']
            else:
                # 回退到基础检测
                plate_regions = self._basic_plate_detection(best_image)
                if not plate_regions:
                    result['details']['error'] = '未检测到车牌区域'
                    return result
                
                largest_region = max(plate_regions, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_region
                plate_img = best_image[y:y+h, x:x+w]
            
            # 多引擎OCR识别
            if self.ocr_engine:
                ocr_results = self.ocr_engine.recognize_multi_engine(plate_img)
                if ocr_results and ocr_results.get('success'):
                    result['success'] = True
                    result['plate_text'] = ocr_results['final_result']
                    result['confidence'] = ocr_results['confidence']
                    result['details'] = ocr_results.get('details', {})
            else:
                # 回退到基础OCR
                plate_text = self._basic_ocr(plate_img)
                if plate_text:
                    result['success'] = True
                    result['plate_text'] = plate_text
                    result['confidence'] = 0.8
            
            return result
            
        except Exception as e:
            result['details']['error'] = str(e)
            return result
    
    def _fusion_recognition(self, image: np.ndarray) -> Dict[str, Any]:
        """融合识别流程（结合多种方法）"""
        result = {
            'success': False,
            'method': 'fusion',
            'plate_text': '',
            'confidence': 0.0,
            'details': {}
        }
        
        try:
            # 并行执行多种识别方法
            methods = []
            
            # 添加基础方法
            methods.append(('basic', lambda: self._basic_recognition(image)))
            
            # 添加高级方法
            if self.image_processor or self.plate_detector or self.ocr_engine:
                methods.append(('advanced', lambda: self._advanced_recognition(image)))
            
            # 执行所有方法
            results = []
            with ThreadPoolExecutor(max_workers=len(methods)) as executor:
                future_to_method = {
                    executor.submit(method_func): method_name 
                    for method_name, method_func in methods
                }
                
                for future in as_completed(future_to_method):
                    method_name = future_to_method[future]
                    try:
                        method_result = future.result(timeout=30)
                        if method_result.get('success'):
                            results.append((method_name, method_result))
                    except Exception as e:
                        self.logger.warning(f"方法 {method_name} 执行失败: {e}")
            
            # 融合结果
            if results:
                # 选择置信度最高的结果
                best_method, best_result = max(results, key=lambda x: x[1].get('confidence', 0))
                
                result['success'] = True
                result['plate_text'] = best_result['plate_text']
                result['confidence'] = best_result['confidence']
                result['details']['best_method'] = best_method
                result['details']['all_results'] = {
                    method: res['plate_text'] for method, res in results
                }
            else:
                result['details']['error'] = '所有识别方法都失败了'
            
            return result
            
        except Exception as e:
            result['details']['error'] = str(e)
            return result
    
    def _basic_plate_detection(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """基础车牌检测"""
        try:
            # 边缘检测
            edges = cv2.Canny(image, 50, 150)
            
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤车牌候选区域
            plate_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                area = w * h
                
                # 车牌尺寸和比例过滤
                if (2.0 < aspect_ratio < 6.0 and 
                    1000 < area < 50000 and
                    w > 80 and h > 20):
                    plate_regions.append((x, y, w, h))
            
            return plate_regions
            
        except Exception as e:
            self.logger.error(f"基础车牌检测失败: {e}")
            return []
    
    def _basic_ocr(self, plate_image: np.ndarray) -> str:
        """基础OCR识别"""
        try:
            import pytesseract
            
            # 图像预处理
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image
            
            # 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR识别
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领'
            text = pytesseract.image_to_string(binary, config=config).strip()
            
            # 清理结果
            text = ''.join(c for c in text if c.isalnum() or c in '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领')
            
            return text if len(text) >= 6 else ''
            
        except Exception as e:
            self.logger.error(f"基础OCR识别失败: {e}")
            return ''
    
    def _generate_cache_key(self, image: np.ndarray) -> str:
        """生成图像缓存键"""
        try:
            # 使用图像的哈希值作为缓存键
            import hashlib
            image_bytes = image.tobytes()
            return hashlib.md5(image_bytes).hexdigest()
        except:
            return None
    
    def _update_performance_metrics(self, result: Dict[str, Any], processing_time: float):
        """更新性能指标"""
        self.performance_metrics['total_processed'] += 1
        
        if result.get('success'):
            self.performance_metrics['success_count'] += 1
        
        # 更新平均处理时间
        total = self.performance_metrics['total_processed']
        current_avg = self.performance_metrics['average_time']
        self.performance_metrics['average_time'] = (current_avg * (total - 1) + processing_time) / total
        
        # 更新准确率
        self.performance_metrics['accuracy_rate'] = (
            self.performance_metrics['success_count'] / self.performance_metrics['total_processed'] * 100
        )
    
    def batch_process(self, image_paths: List[str], method: str = "fusion") -> List[Dict[str, Any]]:
        """批量处理图像"""
        results = []
        
        def process_single(path):
            try:
                image = cv2.imread(path)
                if image is None:
                    return {'error': f'无法加载图像: {path}', 'path': path}
                
                result = self.recognize_plate(image, method)
                result['image_path'] = path
                return result
            except Exception as e:
                return {'error': str(e), 'path': path}
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single, path) for path in image_paths]
            for future in as_completed(futures):
                results.append(future.result())
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'metrics': self.performance_metrics.copy(),
            'cache_size': len(self.result_cache) if self.result_cache else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_system_test(self) -> Dict[str, Any]:
        """运行系统测试"""
        try:
            # 检查是否有测试框架
            if hasattr(self, 'test_suite') or 'test_framework' in globals():
                test_suite = PlateRecognitionTestSuite()
                return test_suite.run_comprehensive_test()
            else:
                # 简单的系统检查
                test_result = {
                    'timestamp': datetime.now().isoformat(),
                    'components': {},
                    'basic_test': {}
                }
                
                # 检查各组件状态
                test_result['components']['image_processor'] = self.image_processor is not None
                test_result['components']['plate_detector'] = self.plate_detector is not None
                test_result['components']['ocr_engine'] = self.ocr_engine is not None
                test_result['components']['crnn_model'] = self.crnn_model is not None
                
                # 创建测试图像
                test_image = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
                test_result['basic_test'] = self.recognize_plate(test_image, "basic")
                
                return test_result
                
        except Exception as e:
            return {'error': f'系统测试失败: {e}'}


def create_demo_usage():
    """创建使用示例"""
    print("=== 优化车牌识别系统使用示例 ===\\n")
    
    # 初始化系统
    print("1. 初始化系统...")
    system = OptimizedPlateRecognitionSystem()
    
    # 创建测试图像
    print("2. 创建测试图像...")
    test_image = np.random.randint(0, 255, (200, 600, 3), dtype=np.uint8)
    
    # 基础识别
    print("3. 基础识别测试...")
    basic_result = system.recognize_plate(test_image, "basic")
    print(f"   基础识别结果: {basic_result['success']}")
    
    # 高级识别
    print("4. 高级识别测试...")
    advanced_result = system.recognize_plate(test_image, "advanced")
    print(f"   高级识别结果: {advanced_result['success']}")
    
    # 融合识别
    print("5. 融合识别测试...")
    fusion_result = system.recognize_plate(test_image, "fusion")
    print(f"   融合识别结果: {fusion_result['success']}")
    
    # 性能报告
    print("6. 性能报告...")
    performance = system.get_performance_report()
    print(f"   处理图像数: {performance['metrics']['total_processed']}")
    print(f"   平均处理时间: {performance['metrics']['average_time']:.3f}秒")
    
    # 系统测试
    print("7. 系统测试...")
    system_test = system.run_system_test()
    print(f"   系统测试完成: {system_test.get('timestamp', 'N/A')}")
    
    print("\\n=== 系统演示完成 ===")


if __name__ == "__main__":
    # 运行演示
    create_demo_usage()
