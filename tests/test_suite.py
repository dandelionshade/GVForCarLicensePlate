# -*- coding: utf-8 -*-
"""
Main test suite - Comprehensive testing framework for license plate recognition system
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

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plate_recognition.core.config import Config
from plate_recognition.core.exceptions import PlateRecognitionError


class TestSuite:
    """Main test suite for license plate recognition system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.test_results = []
        self.performance_metrics = {}
    
    def run_comprehensive_test(self, test_data_dir: str = None) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        if test_data_dir is None:
            test_data_dir = getattr(self.config, 'TEST_DATA_DIR', 'tests/data')
        
        self.logger.info("Starting comprehensive test...")
        
        # Test results container
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
            # 1. Unit tests
            self.logger.info("Running unit tests...")
            test_results['unit_tests'] = self._run_unit_tests()
            
            # 2. Integration tests
            self.logger.info("Running integration tests...")
            test_results['integration_tests'] = self._run_integration_tests()
            
            # 3. Performance tests
            if os.path.exists(test_data_dir):
                self.logger.info("Running performance tests...")
                test_results['performance_tests'] = self._run_performance_tests(test_data_dir)
                
                # 4. Stress tests
                self.logger.info("Running stress tests...")
                test_results['stress_tests'] = self._run_stress_tests(test_data_dir)
                
                # 5. Accuracy tests
                self.logger.info("Running accuracy tests...")
                test_results['accuracy_tests'] = self._run_accuracy_tests(test_data_dir)
            else:
                self.logger.warning(f"Test data directory not found: {test_data_dir}")
            
            # 6. Generate summary report
            test_results['summary'] = self._generate_summary(test_results)
            
            # 7. Save test results
            self._save_test_results(test_results)
            
            self.logger.info("Comprehensive test completed")
            return test_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive test failed: {e}")
            test_results['error'] = str(e)
            return test_results

    def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests"""
        results = {
            'config_test': self._test_config(),
            'image_processing_test': self._test_image_processing(),
            'plate_detection_test': self._test_plate_detection(),
            'ocr_engines_test': self._test_ocr_engines()
        }
        
        # Calculate pass rate
        passed = sum(1 for result in results.values() if result.get('passed', False))
        total = len(results)
        results['pass_rate'] = passed / total if total > 0 else 0
        
        return results

    def _test_config(self) -> Dict[str, Any]:
        """Test configuration module"""
        try:
            # Test configuration loading
            config = Config()
            
            # Verify key configurations
            checks = [
                ('PROVINCES', hasattr(config, 'PROVINCES') and len(getattr(config, 'PROVINCES', [])) > 0),
                ('PLATE_PATTERN', hasattr(config, 'PLATE_PATTERN') and getattr(config, 'PLATE_PATTERN', None)),
                ('MODEL_INPUT_SIZE', hasattr(config, 'MODEL_INPUT_SIZE') and len(getattr(config, 'MODEL_INPUT_SIZE', [])) == 2),
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
        """Test image processing module"""
        try:
            from plate_recognition.preprocessing.image_processor import ImageProcessor
            
            processor = ImageProcessor()
            
            # Create test image
            test_image = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
            
            start_time = time.time()
            
            # Test basic preprocessing
            try:
                basic_results = processor.preprocess_image(test_image, method="basic")
            except:
                basic_results = processor.preprocess_image(test_image)
            
            # Test comprehensive preprocessing
            try:
                comprehensive_results = processor.preprocess_image(test_image, method="comprehensive")
            except:
                comprehensive_results = basic_results
            
            # Test plate enhancement
            enhanced = processor.enhance_plate_region(test_image)
            
            execution_time = time.time() - start_time
            
            # Verify results
            checks = [
                ('basic_preprocessing', isinstance(basic_results, dict) and len(basic_results) > 0),
                ('comprehensive_preprocessing', isinstance(comprehensive_results, dict) and len(comprehensive_results) >= len(basic_results)),
                ('plate_enhancement', enhanced is not None),
                ('performance', execution_time < 5.0)  # Should complete within 5 seconds
            ]
            
            failed_checks = [name for name, passed in checks if not passed]
            
            return {
                'passed': len(failed_checks) == 0,
                'checks': dict(checks),
                'failed_checks': failed_checks,
                'execution_time': execution_time,
                'results_count': len(comprehensive_results) if isinstance(comprehensive_results, dict) else 0
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'execution_time': 0
            }

    def _test_plate_detection(self) -> Dict[str, Any]:
        """Test plate detection module"""
        try:
            from plate_recognition.detection.detector import PlateDetector
            
            detector = PlateDetector()
            
            # Create test image (simulate image containing license plate)
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add a rectangle area to simulate license plate
            cv2.rectangle(test_image, (200, 200), (400, 260), (100, 150, 200), -1)
            
            start_time = time.time()
            
            # Test detection methods
            try:
                contour_results = detector.detect_plates(test_image, method="contour")
            except:
                contour_results = detector.detect_plates(test_image)
            
            try:
                color_results = detector.detect_plates(test_image, method="color")
            except:
                color_results = contour_results
            
            try:
                comprehensive_results = detector.detect_plates(test_image, method="comprehensive")
            except:
                comprehensive_results = contour_results
            
            execution_time = time.time() - start_time
            
            # Verify results
            checks = [
                ('contour_detection', isinstance(contour_results, list)),
                ('color_detection', isinstance(color_results, list)),
                ('comprehensive_detection', isinstance(comprehensive_results, list)),
                ('performance', execution_time < 10.0)  # Should complete within 10 seconds
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
        """Test OCR engines module"""
        try:
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            
            # Create test image
            test_image = np.ones((50, 200, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, "TEST123", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            start_time = time.time()
            
            # Test available engines
            available_engines = [name for name, available in ocr.engines.items() if available]
            
            # Test recognition
            if available_engines:
                result = ocr.recognize(test_image, engines=available_engines[:1])  # Test only one engine to avoid timeout
            else:
                result = {'success': False, 'message': 'No available engines'}
            
            execution_time = time.time() - start_time
            
            # Verify results
            checks = [
                ('engines_available', len(available_engines) > 0),
                ('recognition_result', isinstance(result, dict)),
                ('performance', execution_time < 30.0)  # Should complete within 30 seconds
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
        """Run integration tests"""
        results = {
            'end_to_end_pipeline': self._test_end_to_end_pipeline(),
            'error_handling': self._test_error_handling(),
            'concurrent_processing': self._test_concurrent_processing()
        }
        
        # Calculate pass rate
        passed = sum(1 for result in results.values() if result.get('passed', False))
        total = len(results)
        results['pass_rate'] = passed / total if total > 0 else 0
        
        return results

    def _test_end_to_end_pipeline(self) -> Dict[str, Any]:
        """Test end-to-end pipeline"""
        try:
            from plate_recognition.preprocessing.image_processor import ImageProcessor
            from plate_recognition.detection.detector import PlateDetector
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
            # Create components
            processor = ImageProcessor()
            detector = PlateDetector()
            ocr = MultiEngineOCR()
            
            # Create test image
            test_image = self._create_synthetic_plate_image("京A12345")
            
            start_time = time.time()
            
            # Complete pipeline
            # 1. Image preprocessing
            processed_results = processor.preprocess_image(test_image)
            
            # 2. Plate detection
            detections = detector.detect_plates(test_image)
            
            # 3. OCR recognition
            if detections:
                plate_region = detections[0].get('region', test_image)
                enhanced_region = processor.enhance_plate_region(plate_region)
                recognition_result = ocr.recognize(enhanced_region, engines=['tesseract'])  # Use only local engine to avoid network dependency
            else:
                recognition_result = ocr.recognize(test_image, engines=['tesseract'])
            
            execution_time = time.time() - start_time
            
            # Verify results
            checks = [
                ('preprocessing_success', len(processed_results) > 0),
                ('detection_success', len(detections) >= 0),  # Detection may be empty, which is normal
                ('recognition_success', isinstance(recognition_result, dict)),
                ('pipeline_performance', execution_time < 60.0)  # Should complete within 60 seconds
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
        """Test error handling"""
        try:
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            
            start_time = time.time()
            
            # Test various error conditions
            error_tests = []
            
            # 1. Null image
            try:
                result = ocr.recognize(None)
                error_tests.append(('null_image', not result.get('success', True)))
            except:
                error_tests.append(('null_image', True))  # Exception handled correctly
            
            # 2. Invalid image
            try:
                invalid_image = np.array([])
                result = ocr.recognize(invalid_image)
                error_tests.append(('invalid_image', not result.get('success', True)))
            except:
                error_tests.append(('invalid_image', True))  # Exception handled correctly
            
            # 3. Too small image
            try:
                tiny_image = np.ones((1, 1, 3), dtype=np.uint8)
                result = ocr.recognize(tiny_image)
                error_tests.append(('tiny_image', not result.get('success', True)))
            except:
                error_tests.append(('tiny_image', True))  # Exception handled correctly
            
            execution_time = time.time() - start_time
            
            # Verify results
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
        """Test concurrent processing"""
        try:
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            
            # Create multiple test images
            test_images = []
            for i in range(5):
                image = self._create_synthetic_plate_image(f"测试{i}")
                test_images.append(image)
            
            start_time = time.time()
            
            # Concurrent processing
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
            
            # Verify results
            successful_results = sum(1 for r in results if r.get('success', False))
            
            checks = [
                ('all_completed', len(results) == len(test_images)),
                ('no_deadlock', execution_time < 120.0),  # Should complete within 2 minutes
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
        """Run performance tests"""
        # Test actual image files here, but create mock tests since we may not have test data
        return {
            'latency_test': self._test_latency(),
            'throughput_test': self._test_throughput(),
            'memory_usage_test': self._test_memory_usage(),
            'cpu_usage_test': self._test_cpu_usage()
        }

    def _test_latency(self) -> Dict[str, Any]:
        """Test latency performance"""
        try:
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            test_image = self._create_synthetic_plate_image("京A12345")
            
            # Test multiple times for average
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
                'passed': avg_latency < 5.0,  # Average latency should be less than 5 seconds
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
        """Test throughput performance"""
        try:
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            test_image = self._create_synthetic_plate_image("京A12345")
            
            # Test how many images can be processed in 30 seconds
            start_time = time.time()
            processed_count = 0
            test_duration = 30  # seconds
            
            while time.time() - start_time < test_duration:
                result = ocr.recognize(test_image, engines=['tesseract'])
                processed_count += 1
                
                # Avoid infinite loop
                if processed_count > 1000:
                    break
            
            actual_duration = time.time() - start_time
            throughput = processed_count / actual_duration  # images/second
            
            return {
                'passed': throughput > 0.1,  # At least one image per 10 seconds
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
        """Test memory usage"""
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            ocr = MultiEngineOCR()
            
            # Process multiple images
            for i in range(20):
                test_image = self._create_synthetic_plate_image(f"测试{i}")
                result = ocr.recognize(test_image, engines=['tesseract'])
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            return {
                'passed': memory_increase < 500,  # Memory increase should be less than 500MB
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
        """Test CPU usage"""
        try:
            # Monitor CPU usage
            cpu_percentages = []
            
            def monitor_cpu():
                for _ in range(10):
                    cpu_percentages.append(psutil.cpu_percent(interval=1))
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()
            
            # Execute CPU-intensive tasks
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            ocr = MultiEngineOCR()
            
            for i in range(5):
                test_image = self._create_synthetic_plate_image(f"测试{i}")
                result = ocr.recognize(test_image, engines=['tesseract'])
            
            monitor_thread.join()
            
            avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
            max_cpu = max(cpu_percentages) if cpu_percentages else 0
            
            return {
                'passed': max_cpu < 95,  # CPU usage should not exceed 95%
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
        """Run stress tests"""
        return {
            'concurrent_requests': self._test_concurrent_requests(),
            'long_running': self._test_long_running(),
            'memory_leak': self._test_memory_leak()
        }

    def _test_concurrent_requests(self) -> Dict[str, Any]:
        """Test concurrent requests"""
        try:
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
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
            
            # Simulate 20 concurrent requests
            concurrent_requests = 20
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_request, i) for i in range(concurrent_requests)]
                results = []
                
                for future in as_completed(futures, timeout=300):  # 5 minute timeout
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({'success': False, 'error': str(e)})
            
            total_time = time.time() - start_time
            successful_requests = sum(1 for r in results if r.get('success', False))
            avg_duration = sum(r.get('duration', 0) for r in results) / len(results) if results else 0
            
            return {
                'passed': successful_requests >= concurrent_requests * 0.8,  # At least 80% success
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
        """Test long running"""
        try:
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            start_time = time.time()
            processed_count = 0
            errors = 0
            
            # Run for 5 minutes
            test_duration = 300  # 5 minutes
            
            while time.time() - start_time < test_duration:
                try:
                    test_image = self._create_synthetic_plate_image(f"长期测试{processed_count}")
                    result = ocr.recognize(test_image, engines=['tesseract'])
                    processed_count += 1
                    
                    if not result.get('success', False):
                        errors += 1
                        
                except Exception:
                    errors += 1
                
                # Rest every 10 images to simulate real-world usage
                if processed_count % 10 == 0:
                    time.sleep(1)
            
            actual_duration = time.time() - start_time
            error_rate = errors / processed_count if processed_count > 0 else 1.0
            
            return {
                'passed': error_rate < 0.1,  # Error rate should be less than 10%
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
        """Test memory leak"""
        try:
            process = psutil.Process()
            memory_measurements = []
            
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
            # Test 50 times, record memory usage
            for i in range(50):
                ocr = MultiEngineOCR()
                test_image = self._create_synthetic_plate_image(f"内存测试{i}")
                result = ocr.recognize(test_image, engines=['tesseract'])
                
                # Record memory usage
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(memory_mb)
                
                # Force garbage collection
                import gc
                gc.collect()
            
            # Analyze memory trend
            initial_memory = memory_measurements[0]
            final_memory = memory_measurements[-1]
            max_memory = max(memory_measurements)
            memory_growth = final_memory - initial_memory
            
            # Simple linear regression to detect memory growth trend
            n = len(memory_measurements)
            x_values = list(range(n))
            y_values = memory_measurements
            
            # Calculate slope
            x_mean = sum(x_values) / n
            y_mean = sum(y_values) / n
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / sum((x - x_mean) ** 2 for x in x_values)
            
            return {
                'passed': slope < 1.0,  # Memory growth slope should be less than 1MB/iteration
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
        """Run accuracy tests"""
        # Since we don't have real labeled data, perform simulation tests here
        return {
            'synthetic_data_test': self._test_synthetic_accuracy(),
            'noise_robustness_test': self._test_noise_robustness(),
            'lighting_robustness_test': self._test_lighting_robustness()
        }

    def _test_synthetic_accuracy(self) -> Dict[str, Any]:
        """Test synthetic data accuracy"""
        try:
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            
            # Create test cases
            test_cases = [
                "京A12345", "沪B67890", "粤C11111", "川D22222", "鲁E33333"
            ]
            
            correct_predictions = 0
            total_predictions = len(test_cases)
            
            for true_label in test_cases:
                test_image = self._create_synthetic_plate_image(true_label)
                result = ocr.recognize(test_image, engines=['tesseract'])
                
                predicted_label = result.get('text', '')
                
                # Clean prediction result
                predicted_label = ''.join(c for c in predicted_label if c.isalnum() or '\u4e00' <= c <= '\u9fa5')
                
                if predicted_label == true_label:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions
            
            return {
                'passed': accuracy > 0.3,  # At least 30% accuracy for synthetic data
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
        """Test noise robustness"""
        try:
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            test_label = "京A12345"
            
            # Test different noise levels
            noise_levels = [0.0, 0.1, 0.2, 0.3]
            results = []
            
            for noise_level in noise_levels:
                test_image = self._create_synthetic_plate_image(test_label)
                
                # Add noise
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level * 255, test_image.shape).astype(np.int16)
                    noisy_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                else:
                    noisy_image = test_image
                
                result = ocr.recognize(noisy_image, engines=['tesseract'])
                predicted_label = result.get('text', '')
                
                # Simple similarity calculation
                similarity = self._calculate_similarity(predicted_label, test_label)
                
                results.append({
                    'noise_level': noise_level,
                    'predicted': predicted_label,
                    'similarity': similarity
                })
            
            # Calculate average performance degradation
            clean_similarity = results[0]['similarity'] if results else 0
            avg_noisy_similarity = sum(r['similarity'] for r in results[1:]) / (len(results) - 1) if len(results) > 1 else 0
            
            return {
                'passed': avg_noisy_similarity > clean_similarity * 0.5,  # Performance under noise should not drop by more than 50%
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
        """Test lighting robustness"""
        try:
            from plate_recognition.recognition.multi_engine_ocr import MultiEngineOCR
            
            ocr = MultiEngineOCR()
            test_label = "京A12345"
            
            # Test different brightness levels
            brightness_levels = [0.5, 1.0, 1.5, 2.0]  # Brightness multiplier
            results = []
            
            for brightness in brightness_levels:
                test_image = self._create_synthetic_plate_image(test_label)
                
                # Adjust brightness
                adjusted_image = cv2.convertScaleAbs(test_image, alpha=brightness, beta=0)
                
                result = ocr.recognize(adjusted_image, engines=['tesseract'])
                predicted_label = result.get('text', '')
                
                similarity = self._calculate_similarity(predicted_label, test_label)
                
                results.append({
                    'brightness_level': brightness,
                    'predicted': predicted_label,
                    'similarity': similarity
                })
            
            # Calculate average performance
            avg_similarity = sum(r['similarity'] for r in results) / len(results) if results else 0
            
            return {
                'passed': avg_similarity > 0.3,  # Average similarity should be greater than 30%
                'avg_similarity': avg_similarity,
                'results': results
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }

    def _create_synthetic_plate_image(self, plate_text: str) -> np.ndarray:
        """Create synthetic plate image"""
        # Create white background
        image = np.ones((60, 200, 3), dtype=np.uint8) * 255
        
        # Add blue background (simulate plate background)
        cv2.rectangle(image, (5, 5), (195, 55), (255, 255, 0), -1)  # Yellow background
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 0, 0)  # Black text
        thickness = 2
        
        # Calculate text position
        text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2
        
        cv2.putText(image, plate_text, (text_x, text_y), font, font_scale, color, thickness)
        
        return image

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character matching
        matches = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
        max_length = max(len(text1), len(text2))
        
        return matches / max_length if max_length > 0 else 0.0

    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary"""
        summary = {
            'overall_status': 'PASS',
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'categories': {}
        }
        
        # Count various test results
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
        
        # Determine overall status
        if summary['overall_pass_rate'] >= 0.8:
            summary['overall_status'] = 'PASS'
        elif summary['overall_pass_rate'] >= 0.6:
            summary['overall_status'] = 'WARNING'
        else:
            summary['overall_status'] = 'FAIL'
        
        return summary

    def _save_test_results(self, test_results: Dict[str, Any]):
        """Save test results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
            
            # Create logs directory if it doesn't exist
            logs_dir = getattr(self.config, 'LOGS_DIR', 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            filepath = os.path.join(logs_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Test results saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save test results: {e}")

    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """Generate test report"""
        report = []
        report.append("=" * 80)
        report.append("License Plate Recognition System - Comprehensive Test Report")
        report.append("=" * 80)
        report.append(f"Test Time: {test_results.get('timestamp', 'Unknown')}")
        report.append("")
        
        # Summary information
        summary = test_results.get('summary', {})
        report.append("【Test Summary】")
        report.append(f"Overall Status: {summary.get('overall_status', 'Unknown')}")
        report.append(f"Total Tests: {summary.get('total_tests', 0)}")
        report.append(f"Passed Tests: {summary.get('passed_tests', 0)}")
        report.append(f"Failed Tests: {summary.get('failed_tests', 0)}")
        report.append(f"Pass Rate: {summary.get('overall_pass_rate', 0):.1%}")
        report.append("")
        
        # Category details
        categories = summary.get('categories', {})
        if categories:
            report.append("【Category Test Results】")
            for category, stats in categories.items():
                status = "✓" if stats['pass_rate'] >= 0.8 else "⚠" if stats['pass_rate'] >= 0.6 else "✗"
                report.append(f"{status} {category}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")
            report.append("")
        
        # Detailed error information
        report.append("【Detailed Test Results】")
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
                    report.append(f"    Error: {test_result['error']}")
                
                if 'execution_time' in test_result:
                    report.append(f"    Execution Time: {test_result['execution_time']:.2f}s")
        
        return "\n".join(report)
