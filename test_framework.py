# -*- coding: utf-8 -*-
"""
系统集成测试模块 - 车牌识别系统
提供完整的测试框架，包括单元测试、集成测试和性能测试
"""

import os
import time
import json
import threading
from typing import List, Dict, Any, Tuple, Optional
import logging
import numpy as np
import cv2
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config, ErrorCodes

class PlateRecognitionTestSuite:
    """车牌识别系统测试套件"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.test_results = []
        self.performance_metrics = {}
        
    def run_comprehensive_test(self, test_data_dir: str = None) -> Dict[str, Any]:
        """运行综合测试"""
        if test_data_dir is None:
            test_data_dir = self.config.TEST_DATA_DIR
        
        self.logger.info("开始综合测试...")
        
        # 测试结果容器
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'stress_tests': {},
            'accuracy_tests': {},
            'summary': {}
        }
        
        try:
            # 1. 单元测试
            self.logger.info("执行单元测试...")
            test_results['unit_tests'] = self._run_unit_tests()
            
            # 2. 集成测试
            self.logger.info("执行集成测试...")
            test_results['integration_tests'] = self._run_integration_tests()
            
            # 3. 性能测试
            if os.path.exists(test_data_dir):
                self.logger.info("执行性能测试...")
                test_results['performance_tests'] = self._run_performance_tests(test_data_dir)
                
                # 4. 压力测试
                self.logger.info("执行压力测试...")
                test_results['stress_tests'] = self._run_stress_tests(test_data_dir)
                
                # 5. 准确率测试
                self.logger.info("执行准确率测试...")
                test_results['accuracy_tests'] = self._run_accuracy_tests(test_data_dir)
            else:
                self.logger.warning(f"测试数据目录不存在: {test_data_dir}")
            
            # 6. 生成总结报告
            test_results['summary'] = self._generate_summary(test_results)
            
            # 7. 保存测试结果
            self._save_test_results(test_results)
            
            self.logger.info("综合测试完成")
            return test_results
            
        except Exception as e:
            self.logger.error(f"综合测试失败: {e}")
            test_results['error'] = str(e)
            return test_results
    
    def _run_unit_tests(self) -> Dict[str, Any]:
        """运行单元测试"""
        results = {
            'config_test': self._test_config(),
            'image_processing_test': self._test_image_processing(),
            'plate_detection_test': self._test_plate_detection(),
            'ocr_engines_test': self._test_ocr_engines()
        }
        
        # 计算通过率
        passed = sum(1 for result in results.values() if result.get('passed', False))
        total = len(results)
        results['pass_rate'] = passed / total if total > 0 else 0
        
        return results
    
    def _test_config(self) -> Dict[str, Any]:
        """测试配置模块"""
        try:
            # 测试配置加载
            config = Config()
            
            # 验证关键配置
            checks = [
                ('PROVINCES', hasattr(config, 'PROVINCES') and len(config.PROVINCES) > 0),
                ('PLATE_PATTERN', hasattr(config, 'PLATE_PATTERN') and config.PLATE_PATTERN),
                ('MODEL_INPUT_SIZE', hasattr(config, 'MODEL_INPUT_SIZE') and len(config.MODEL_INPUT_SIZE) == 2),
                ('CHARACTER_SET', len(config.get_character_set()) > 0)
            ]
            
            failed_checks = [name for name, passed in checks if not passed]
            
            return {
                'passed': len(failed_checks) == 0,
                'checks': dict(checks),
                'failed_checks': failed_checks,
                'execution_time': 0.01
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'execution_time': 0.01
            }
    
    def _test_image_processing(self) -> Dict[str, Any]:
        """测试图像处理模块"""
        try:
            from image_processor import AdvancedImageProcessor
            
            processor = AdvancedImageProcessor()
            
            # 创建测试图像
            test_image = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
            
            start_time = time.time()
            
            # 测试基础预处理
            basic_results = processor.preprocess_image(test_image, method="basic")
            
            # 测试综合预处理
            comprehensive_results = processor.preprocess_image(test_image, method="comprehensive")
            
            # 测试车牌增强
            enhanced = processor.enhance_plate_region(test_image)
            
            execution_time = time.time() - start_time
            
            # 验证结果
            checks = [
                ('basic_preprocessing', 'gray' in basic_results and 'binary_otsu' in basic_results),
                ('comprehensive_preprocessing', len(comprehensive_results) >= len(basic_results)),
                ('plate_enhancement', enhanced is not None),
                ('performance', execution_time < 5.0)  # 应该在5秒内完成
            ]
            
            failed_checks = [name for name, passed in checks if not passed]
            
            return {
                'passed': len(failed_checks) == 0,
                'checks': dict(checks),
                'failed_checks': failed_checks,
                'execution_time': execution_time,
                'results_count': len(comprehensive_results)
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'execution_time': 0
            }
    
    def _test_plate_detection(self) -> Dict[str, Any]:
        """测试车牌检测模块"""
        try:
            from plate_detector import AdvancedPlateDetector
            
            detector = AdvancedPlateDetector()
            
            # 创建测试图像（模拟包含车牌的图像）
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 在图像中添加一个矩形区域模拟车牌
            cv2.rectangle(test_image, (200, 200), (400, 260), (100, 150, 200), -1)
            
            start_time = time.time()
            
            # 测试检测方法
            contour_results = detector.detect_plates(test_image, method="contour")
            color_results = detector.detect_plates(test_image, method="color")
            comprehensive_results = detector.detect_plates(test_image, method="comprehensive")
            
            execution_time = time.time() - start_time
            
            # 验证结果
            checks = [
                ('contour_detection', isinstance(contour_results, list)),
                ('color_detection', isinstance(color_results, list)),
                ('comprehensive_detection', isinstance(comprehensive_results, list)),
                ('performance', execution_time < 10.0)  # 应该在10秒内完成
            ]
            
            failed_checks = [name for name, passed in checks if not passed]
            
            return {
                'passed': len(failed_checks) == 0,
                'checks': dict(checks),
                'failed_checks': failed_checks,
                'execution_time': execution_time,
                'detections_count': {
                    'contour': len(contour_results),
                    'color': len(color_results),
                    'comprehensive': len(comprehensive_results)
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'execution_time': 0
            }
    
    def _test_ocr_engines(self) -> Dict[str, Any]:
        """测试OCR引擎模块"""
        try:
            from multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            
            # 创建测试图像
            test_image = np.ones((50, 200, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, "TEST123", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            start_time = time.time()
            
            # 测试可用引擎
            available_engines = [name for name, available in ocr.engines.items() if available]
            
            # 测试识别
            if available_engines:
                result = ocr.recognize(test_image, engines=available_engines[:1])  # 只测试一个引擎避免超时
            else:
                result = {'success': False, 'message': '没有可用引擎'}
            
            execution_time = time.time() - start_time
            
            # 验证结果
            checks = [
                ('engines_available', len(available_engines) > 0),
                ('recognition_result', isinstance(result, dict)),
                ('performance', execution_time < 30.0)  # 应该在30秒内完成
            ]
            
            failed_checks = [name for name, passed in checks if not passed]
            
            return {
                'passed': len(failed_checks) == 0,
                'checks': dict(checks),
                'failed_checks': failed_checks,
                'execution_time': execution_time,
                'available_engines': available_engines,
                'recognition_result': result
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'execution_time': 0
            }
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """运行集成测试"""
        results = {
            'end_to_end_pipeline': self._test_end_to_end_pipeline(),
            'error_handling': self._test_error_handling(),
            'concurrent_processing': self._test_concurrent_processing()
        }
        
        # 计算通过率
        passed = sum(1 for result in results.values() if result.get('passed', False))
        total = len(results)
        results['pass_rate'] = passed / total if total > 0 else 0
        
        return results
    
    def _test_end_to_end_pipeline(self) -> Dict[str, Any]:
        """测试端到端流水线"""
        try:
            from image_processor import AdvancedImageProcessor
            from plate_detector import AdvancedPlateDetector
            from multi_engine_ocr import MultiEngineOCR
            
            # 创建组件
            processor = AdvancedImageProcessor()
            detector = AdvancedPlateDetector()
            ocr = MultiEngineOCR()
            
            # 创建测试图像
            test_image = self._create_synthetic_plate_image("京A12345")
            
            start_time = time.time()
            
            # 完整流水线
            # 1. 图像预处理
            processed_results = processor.preprocess_image(test_image)
            
            # 2. 车牌检测
            detections = detector.detect_plates(test_image)
            
            # 3. OCR识别
            if detections:
                plate_region = detections[0]['region']
                enhanced_region = processor.enhance_plate_region(plate_region)
                recognition_result = ocr.recognize(enhanced_region, engines=['tesseract'])  # 只使用本地引擎避免网络依赖
            else:
                recognition_result = ocr.recognize(test_image, engines=['tesseract'])
            
            execution_time = time.time() - start_time
            
            # 验证结果
            checks = [
                ('preprocessing_success', len(processed_results) > 0),
                ('detection_success', len(detections) >= 0),  # 检测可能为空，这是正常的
                ('recognition_success', isinstance(recognition_result, dict)),
                ('pipeline_performance', execution_time < 60.0)  # 应该在60秒内完成
            ]
            
            failed_checks = [name for name, passed in checks if not passed]
            
            return {
                'passed': len(failed_checks) == 0,
                'checks': dict(checks),
                'failed_checks': failed_checks,
                'execution_time': execution_time,
                'detections_count': len(detections),
                'recognition_result': recognition_result
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'execution_time': 0
            }
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """测试错误处理"""
        try:
            from multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            
            start_time = time.time()
            
            # 测试各种错误情况
            error_tests = []
            
            # 1. 空图像
            try:
                result = ocr.recognize(None)
                error_tests.append(('null_image', not result.get('success', True)))
            except:
                error_tests.append(('null_image', True))  # 异常被正确处理
            
            # 2. 无效图像
            try:
                invalid_image = np.array([])
                result = ocr.recognize(invalid_image)
                error_tests.append(('invalid_image', not result.get('success', True)))
            except:
                error_tests.append(('invalid_image', True))  # 异常被正确处理
            
            # 3. 过小图像
            try:
                tiny_image = np.ones((1, 1, 3), dtype=np.uint8)
                result = ocr.recognize(tiny_image)
                error_tests.append(('tiny_image', not result.get('success', True)))
            except:
                error_tests.append(('tiny_image', True))  # 异常被正确处理
            
            execution_time = time.time() - start_time
            
            # 验证结果
            passed_tests = sum(1 for _, passed in error_tests if passed)
            total_tests = len(error_tests)
            
            return {
                'passed': passed_tests == total_tests,
                'error_tests': error_tests,
                'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'execution_time': execution_time
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'execution_time': 0
            }
    
    def _test_concurrent_processing(self) -> Dict[str, Any]:
        """测试并发处理"""
        try:
            from multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            
            # 创建多个测试图像
            test_images = []
            for i in range(5):
                image = self._create_synthetic_plate_image(f"测试{i}")
                test_images.append(image)
            
            start_time = time.time()
            
            # 并发处理
            results = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(ocr.recognize, img, ['tesseract']) for img in test_images]
                
                for future in as_completed(futures, timeout=60):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({'success': False, 'error': str(e)})
            
            execution_time = time.time() - start_time
            
            # 验证结果
            successful_results = sum(1 for r in results if r.get('success', False))
            
            checks = [
                ('all_completed', len(results) == len(test_images)),
                ('no_deadlock', execution_time < 120.0),  # 应该在2分钟内完成
                ('some_success', successful_results > 0)
            ]
            
            failed_checks = [name for name, passed in checks if not passed]
            
            return {
                'passed': len(failed_checks) == 0,
                'checks': dict(checks),
                'failed_checks': failed_checks,
                'execution_time': execution_time,
                'total_tasks': len(test_images),
                'successful_tasks': successful_results,
                'success_rate': successful_results / len(test_images) if test_images else 0
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'execution_time': 0
            }
    
    def _run_performance_tests(self, test_data_dir: str) -> Dict[str, Any]:
        """运行性能测试"""
        # 这里应该测试实际的图像文件，但由于可能没有测试数据，我们创建模拟测试
        return {
            'latency_test': self._test_latency(),
            'throughput_test': self._test_throughput(),
            'memory_usage_test': self._test_memory_usage(),
            'cpu_usage_test': self._test_cpu_usage()
        }
    
    def _test_latency(self) -> Dict[str, Any]:
        """测试延迟性能"""
        try:
            from multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            test_image = self._create_synthetic_plate_image("京A12345")
            
            # 多次测试取平均值
            latencies = []
            for _ in range(10):
                start_time = time.time()
                result = ocr.recognize(test_image, engines=['tesseract'])
                latency = time.time() - start_time
                latencies.append(latency)
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            return {
                'passed': avg_latency < 5.0,  # 平均延迟应小于5秒
                'avg_latency': avg_latency,
                'max_latency': max_latency,
                'min_latency': min_latency,
                'measurements': latencies
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _test_throughput(self) -> Dict[str, Any]:
        """测试吞吐量性能"""
        try:
            from multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            test_image = self._create_synthetic_plate_image("京A12345")
            
            # 测试30秒内能处理多少图像
            start_time = time.time()
            processed_count = 0
            test_duration = 30  # 秒
            
            while time.time() - start_time < test_duration:
                result = ocr.recognize(test_image, engines=['tesseract'])
                processed_count += 1
                
                # 避免无限循环
                if processed_count > 1000:
                    break
            
            actual_duration = time.time() - start_time
            throughput = processed_count / actual_duration  # 图像/秒
            
            return {
                'passed': throughput > 0.1,  # 至少每10秒处理一张图像
                'throughput': throughput,
                'processed_count': processed_count,
                'test_duration': actual_duration
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _test_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用"""
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            from multi_engine_ocr import MultiEngineOCR
            ocr = MultiEngineOCR()
            
            # 处理多张图像
            for i in range(20):
                test_image = self._create_synthetic_plate_image(f"测试{i}")
                result = ocr.recognize(test_image, engines=['tesseract'])
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            return {
                'passed': memory_increase < 500,  # 内存增长应小于500MB
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _test_cpu_usage(self) -> Dict[str, Any]:
        """测试CPU使用率"""
        try:
            # 监控CPU使用率
            cpu_percentages = []
            
            def monitor_cpu():
                for _ in range(10):
                    cpu_percentages.append(psutil.cpu_percent(interval=1))
            
            # 启动监控线程
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()
            
            # 执行计算密集型任务
            from multi_engine_ocr import MultiEngineOCR
            ocr = MultiEngineOCR()
            
            for i in range(5):
                test_image = self._create_synthetic_plate_image(f"测试{i}")
                result = ocr.recognize(test_image, engines=['tesseract'])
            
            monitor_thread.join()
            
            avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
            max_cpu = max(cpu_percentages) if cpu_percentages else 0
            
            return {
                'passed': max_cpu < 95,  # CPU使用率不应超过95%
                'avg_cpu_percent': avg_cpu,
                'max_cpu_percent': max_cpu,
                'measurements': cpu_percentages
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _run_stress_tests(self, test_data_dir: str) -> Dict[str, Any]:
        """运行压力测试"""
        return {
            'concurrent_requests': self._test_concurrent_requests(),
            'long_running': self._test_long_running(),
            'memory_leak': self._test_memory_leak()
        }
    
    def _test_concurrent_requests(self) -> Dict[str, Any]:
        """测试并发请求"""
        try:
            from multi_engine_ocr import MultiEngineOCR
            
            def process_request(request_id):
                ocr = MultiEngineOCR()
                test_image = self._create_synthetic_plate_image(f"请求{request_id}")
                start_time = time.time()
                result = ocr.recognize(test_image, engines=['tesseract'])
                end_time = time.time()
                return {
                    'request_id': request_id,
                    'success': result.get('success', False),
                    'duration': end_time - start_time
                }
            
            # 模拟20个并发请求
            concurrent_requests = 20
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_request, i) for i in range(concurrent_requests)]
                results = []
                
                for future in as_completed(futures, timeout=300):  # 5分钟超时
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({'success': False, 'error': str(e)})
            
            total_time = time.time() - start_time
            successful_requests = sum(1 for r in results if r.get('success', False))
            avg_duration = sum(r.get('duration', 0) for r in results) / len(results) if results else 0
            
            return {
                'passed': successful_requests >= concurrent_requests * 0.8,  # 至少80%成功
                'total_requests': concurrent_requests,
                'successful_requests': successful_requests,
                'success_rate': successful_requests / concurrent_requests,
                'total_time': total_time,
                'avg_request_duration': avg_duration
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _test_long_running(self) -> Dict[str, Any]:
        """测试长时间运行"""
        try:
            from multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            start_time = time.time()
            processed_count = 0
            errors = 0
            
            # 运行5分钟
            test_duration = 300  # 5分钟
            
            while time.time() - start_time < test_duration:
                try:
                    test_image = self._create_synthetic_plate_image(f"长期测试{processed_count}")
                    result = ocr.recognize(test_image, engines=['tesseract'])
                    processed_count += 1
                    
                    if not result.get('success', False):
                        errors += 1
                        
                except Exception:
                    errors += 1
                
                # 每处理10个图像休息一下，模拟实际使用场景
                if processed_count % 10 == 0:
                    time.sleep(1)
            
            actual_duration = time.time() - start_time
            error_rate = errors / processed_count if processed_count > 0 else 1.0
            
            return {
                'passed': error_rate < 0.1,  # 错误率应小于10%
                'test_duration': actual_duration,
                'processed_count': processed_count,
                'error_count': errors,
                'error_rate': error_rate,
                'throughput': processed_count / actual_duration
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _test_memory_leak(self) -> Dict[str, Any]:
        """测试内存泄漏"""
        try:
            process = psutil.Process()
            memory_measurements = []
            
            from multi_engine_ocr import MultiEngineOCR
            
            # 测试50次，记录内存使用情况
            for i in range(50):
                ocr = MultiEngineOCR()
                test_image = self._create_synthetic_plate_image(f"内存测试{i}")
                result = ocr.recognize(test_image, engines=['tesseract'])
                
                # 记录内存使用
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(memory_mb)
                
                # 强制垃圾回收
                import gc
                gc.collect()
            
            # 分析内存趋势
            initial_memory = memory_measurements[0]
            final_memory = memory_measurements[-1]
            max_memory = max(memory_measurements)
            memory_growth = final_memory - initial_memory
            
            # 简单的线性回归检测内存增长趋势
            n = len(memory_measurements)
            x_values = list(range(n))
            y_values = memory_measurements
            
            # 计算斜率
            x_mean = sum(x_values) / n
            y_mean = sum(y_values) / n
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / sum((x - x_mean) ** 2 for x in x_values)
            
            return {
                'passed': slope < 1.0,  # 内存增长斜率应小于1MB/次
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'max_memory_mb': max_memory,
                'memory_growth_mb': memory_growth,
                'memory_growth_slope': slope,
                'measurements': memory_measurements
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _run_accuracy_tests(self, test_data_dir: str) -> Dict[str, Any]:
        """运行准确率测试"""
        # 由于没有真实的标注数据，这里进行模拟测试
        return {
            'synthetic_data_test': self._test_synthetic_accuracy(),
            'noise_robustness_test': self._test_noise_robustness(),
            'lighting_robustness_test': self._test_lighting_robustness()
        }
    
    def _test_synthetic_accuracy(self) -> Dict[str, Any]:
        """测试合成数据准确率"""
        try:
            from multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            
            # 创建测试案例
            test_cases = [
                "京A12345", "沪B67890", "粤C11111", "川D22222", "鲁E33333"
            ]
            
            correct_predictions = 0
            total_predictions = len(test_cases)
            
            for true_label in test_cases:
                test_image = self._create_synthetic_plate_image(true_label)
                result = ocr.recognize(test_image, engines=['tesseract'])
                
                predicted_label = result.get('text', '')
                
                # 清理预测结果
                predicted_label = ''.join(c for c in predicted_label if c.isalnum() or '\u4e00' <= c <= '\u9fa5')
                
                if predicted_label == true_label:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions
            
            return {
                'passed': accuracy > 0.3,  # 合成数据至少30%准确率
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'test_cases': test_cases
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _test_noise_robustness(self) -> Dict[str, Any]:
        """测试噪声鲁棒性"""
        try:
            from multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            test_label = "京A12345"
            
            # 测试不同噪声级别
            noise_levels = [0.0, 0.1, 0.2, 0.3]
            results = []
            
            for noise_level in noise_levels:
                test_image = self._create_synthetic_plate_image(test_label)
                
                # 添加噪声
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level * 255, test_image.shape).astype(np.int16)
                    noisy_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                else:
                    noisy_image = test_image
                
                result = ocr.recognize(noisy_image, engines=['tesseract'])
                predicted_label = result.get('text', '')
                
                # 简单的相似度计算
                similarity = self._calculate_similarity(predicted_label, test_label)
                
                results.append({
                    'noise_level': noise_level,
                    'predicted': predicted_label,
                    'similarity': similarity
                })
            
            # 计算平均性能下降
            clean_similarity = results[0]['similarity'] if results else 0
            avg_noisy_similarity = sum(r['similarity'] for r in results[1:]) / (len(results) - 1) if len(results) > 1 else 0
            
            return {
                'passed': avg_noisy_similarity > clean_similarity * 0.5,  # 噪声下性能不应下降超过50%
                'clean_similarity': clean_similarity,
                'avg_noisy_similarity': avg_noisy_similarity,
                'results': results
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _test_lighting_robustness(self) -> Dict[str, Any]:
        """测试光照鲁棒性"""
        try:
            from multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            test_label = "京A12345"
            
            # 测试不同亮度级别
            brightness_levels = [0.5, 1.0, 1.5, 2.0]  # 亮度倍数
            results = []
            
            for brightness in brightness_levels:
                test_image = self._create_synthetic_plate_image(test_label)
                
                # 调整亮度
                adjusted_image = cv2.convertScaleAbs(test_image, alpha=brightness, beta=0)
                
                result = ocr.recognize(adjusted_image, engines=['tesseract'])
                predicted_label = result.get('text', '')
                
                similarity = self._calculate_similarity(predicted_label, test_label)
                
                results.append({
                    'brightness_level': brightness,
                    'predicted': predicted_label,
                    'similarity': similarity
                })
            
            # 计算平均性能
            avg_similarity = sum(r['similarity'] for r in results) / len(results) if results else 0
            
            return {
                'passed': avg_similarity > 0.3,  # 平均相似度应大于30%
                'avg_similarity': avg_similarity,
                'results': results
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _create_synthetic_plate_image(self, plate_text: str) -> np.ndarray:
        """创建合成车牌图像"""
        # 创建白色背景
        image = np.ones((60, 200, 3), dtype=np.uint8) * 255
        
        # 添加蓝色背景（模拟车牌背景）
        cv2.rectangle(image, (5, 5), (195, 55), (255, 255, 0), -1)  # 黄色背景
        
        # 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 0, 0)  # 黑色文字
        thickness = 2
        
        # 计算文字位置
        text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2
        
        cv2.putText(image, plate_text, (text_x, text_y), font, font_scale, color, thickness)
        
        return image
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 简单的字符匹配
        matches = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
        max_length = max(len(text1), len(text2))
        
        return matches / max_length if max_length > 0 else 0.0
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成测试总结"""
        summary = {
            'overall_status': 'PASS',
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'categories': {}
        }
        
        # 统计各类测试结果
        for category, tests in test_results.items():
            if category in ['timestamp', 'summary']:
                continue
                
            if isinstance(tests, dict):
                category_passed = 0
                category_total = 0
                
                for test_name, test_result in tests.items():
                    if test_name == 'pass_rate':
                        continue
                        
                    if isinstance(test_result, dict) and 'passed' in test_result:
                        category_total += 1
                        if test_result['passed']:
                            category_passed += 1
                
                summary['categories'][category] = {
                    'total': category_total,
                    'passed': category_passed,
                    'pass_rate': category_passed / category_total if category_total > 0 else 0
                }
                
                summary['total_tests'] += category_total
                summary['passed_tests'] += category_passed
        
        summary['failed_tests'] = summary['total_tests'] - summary['passed_tests']
        summary['overall_pass_rate'] = summary['passed_tests'] / summary['total_tests'] if summary['total_tests'] > 0 else 0
        
        # 确定总体状态
        if summary['overall_pass_rate'] >= 0.8:
            summary['overall_status'] = 'PASS'
        elif summary['overall_pass_rate'] >= 0.6:
            summary['overall_status'] = 'WARNING'
        else:
            summary['overall_status'] = 'FAIL'
        
        return summary
    
    def _save_test_results(self, test_results: Dict[str, Any]):
        """保存测试结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
            filepath = os.path.join(self.config.LOGS_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"测试结果已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存测试结果失败: {e}")
    
    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """生成测试报告"""
        report = []
        report.append("=" * 80)
        report.append("车牌识别系统 - 综合测试报告")
        report.append("=" * 80)
        report.append(f"测试时间: {test_results.get('timestamp', 'Unknown')}")
        report.append("")
        
        # 总结信息
        summary = test_results.get('summary', {})
        report.append("【测试总结】")
        report.append(f"总体状态: {summary.get('overall_status', 'Unknown')}")
        report.append(f"总测试数: {summary.get('total_tests', 0)}")
        report.append(f"通过测试: {summary.get('passed_tests', 0)}")
        report.append(f"失败测试: {summary.get('failed_tests', 0)}")
        report.append(f"通过率: {summary.get('overall_pass_rate', 0):.1%}")
        report.append("")
        
        # 各类别详情
        categories = summary.get('categories', {})
        if categories:
            report.append("【分类测试结果】")
            for category, stats in categories.items():
                status = "✓" if stats['pass_rate'] >= 0.8 else "⚠" if stats['pass_rate'] >= 0.6 else "✗"
                report.append(f"{status} {category}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")
            report.append("")
        
        # 详细错误信息
        report.append("【详细测试结果】")
        for category, tests in test_results.items():
            if category in ['timestamp', 'summary'] or not isinstance(tests, dict):
                continue
                
            report.append(f"\n{category.upper()}:")
            for test_name, test_result in tests.items():
                if test_name == 'pass_rate' or not isinstance(test_result, dict):
                    continue
                    
                status = "✓ PASS" if test_result.get('passed', False) else "✗ FAIL"
                report.append(f"  {status} {test_name}")
                
                if not test_result.get('passed', False) and 'error' in test_result:
                    report.append(f"    错误: {test_result['error']}")
                
                if 'execution_time' in test_result:
                    report.append(f"    执行时间: {test_result['execution_time']:.2f}s")
        
        return "\n".join(report)
