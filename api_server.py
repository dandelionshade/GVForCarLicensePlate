# -*- coding: utf-8 -*-
"""
Flask API服务器 - 车牌识别系统
提供RESTful API接口，支持云端部署和远程调用
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import cv2
import base64
from io import BytesIO

try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logging.warning("Flask不可用，API服务功能将被禁用")

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus客户端不可用，监控功能将被禁用")

from config import Config, ErrorCodes

# 条件导入我们的模块
try:
    from image_processor import AdvancedImageProcessor
    from plate_detector import AdvancedPlateDetector
    from multi_engine_ocr import MultiEngineOCR
    from crnn_model import CRNNModel
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logging.warning(f"模块导入失败: {e}")

class PlateRecognitionAPI:
    """车牌识别API服务"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        if not FLASK_AVAILABLE:
            self.logger.error("Flask不可用，无法启动API服务")
            self.app = None
            return
        
        # 初始化Flask应用
        self.app = Flask(__name__)
        CORS(self.app)  # 启用跨域支持
        
        # 初始化识别组件
        if MODULES_AVAILABLE:
            self.image_processor = AdvancedImageProcessor()
            self.plate_detector = AdvancedPlateDetector()
            self.ocr_engine = MultiEngineOCR()
            self.crnn_model = CRNNModel()
        else:
            self.logger.error("识别模块不可用")
            self.image_processor = None
            self.plate_detector = None
            self.ocr_engine = None
            self.crnn_model = None
        
        # 初始化监控指标
        if PROMETHEUS_AVAILABLE:
            self.metrics = {
                'requests_total': Counter('plate_recognition_requests_total', 'Total API requests', ['method', 'status']),
                'request_duration': Histogram('plate_recognition_request_duration_seconds', 'Request duration'),
                'active_requests': Gauge('plate_recognition_active_requests', 'Active requests'),
                'recognition_accuracy': Gauge('plate_recognition_accuracy', 'Recognition accuracy'),
                'errors_total': Counter('plate_recognition_errors_total', 'Total errors', ['error_type'])
            }
        else:
            self.metrics = None
        
        # 注册路由
        self._register_routes()
        
        # 请求统计
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'start_time': datetime.now()
        }
    
    def _register_routes(self):
        """注册API路由"""
        if not self.app:
            return
        
        @self.app.route('/', methods=['GET'])
        def index():
            """API主页"""
            return self._render_api_docs()
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return self._health_check()
        
        @self.app.route('/recognize', methods=['POST'])
        def recognize_plate():
            """车牌识别API"""
            return self._recognize_plate()
        
        @self.app.route('/recognize/batch', methods=['POST'])
        def batch_recognize():
            """批量车牌识别API"""
            return self._batch_recognize()
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_image():
            """图像分析API（包含详细的中间结果）"""
            return self._analyze_image()
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """获取服务统计信息"""
            return self._get_stats()
        
        if PROMETHEUS_AVAILABLE:
            @self.app.route('/metrics', methods=['GET'])
            def metrics():
                """Prometheus监控指标"""
                return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    
    def _render_api_docs(self):
        """渲染API文档"""
        docs_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>车牌识别API服务</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
                .method { background: #4CAF50; color: white; padding: 5px 10px; border-radius: 3px; }
                .method.post { background: #2196F3; }
                .method.get { background: #FF9800; }
                code { background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>车牌识别API服务</h1>
            <p>提供高性能的车牌识别服务，支持多种识别引擎和详细的分析结果。</p>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /health</h3>
                <p>服务健康检查</p>
                <p><strong>响应示例:</strong></p>
                <pre><code>{
  "status": "healthy",
  "timestamp": "2025-07-08T10:30:00",
  "uptime": 3600,
  "components": {...}
}</code></pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /recognize</h3>
                <p>车牌识别</p>
                <p><strong>请求参数:</strong></p>
                <ul>
                    <li><code>image</code> - 图像文件或base64编码的图像数据</li>
                    <li><code>engines</code> - 可选，指定使用的OCR引擎 (默认使用所有可用引擎)</li>
                    <li><code>enhance</code> - 可选，是否启用图像增强 (默认true)</li>
                </ul>
                <p><strong>响应示例:</strong></p>
                <pre><code>{
  "success": true,
  "plate_number": "京A12345",
  "confidence": 0.95,
  "processing_time": 1.23,
  "engine_used": "paddle",
  "detection_info": {...}
}</code></pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /recognize/batch</h3>
                <p>批量车牌识别</p>
                <p><strong>请求参数:</strong></p>
                <ul>
                    <li><code>images</code> - 图像文件数组或base64编码的图像数据数组</li>
                </ul>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /analyze</h3>
                <p>详细图像分析（包含所有中间处理结果）</p>
                <p>返回预处理、检测、识别的详细信息，用于调试和优化。</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /stats</h3>
                <p>获取服务统计信息</p>
                <p>包含请求数量、成功率、平均响应时间等信息。</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /metrics</h3>
                <p>Prometheus监控指标</p>
                <p>提供用于监控系统的Prometheus格式指标。</p>
            </div>
            
            <h2>使用示例</h2>
            <h3>cURL</h3>
            <pre><code>curl -X POST http://localhost:5000/recognize \\
  -F "image=@plate_image.jpg" \\
  -F "engines=[\"paddle\", \"tesseract\"]"</code></pre>
            
            <h3>Python</h3>
            <pre><code>import requests

url = "http://localhost:5000/recognize"
files = {"image": open("plate_image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()
print(result["plate_number"])</code></pre>
            
        </body>
        </html>
        """
        return docs_html
    
    def _health_check(self) -> Dict[str, Any]:
        """健康检查"""
        uptime = (datetime.now() - self.request_stats['start_time']).total_seconds()
        
        components = {}
        
        if MODULES_AVAILABLE:
            components['image_processor'] = self.image_processor is not None
            components['plate_detector'] = self.plate_detector is not None
            components['ocr_engine'] = self.ocr_engine is not None
            components['crnn_model'] = self.crnn_model is not None
        else:
            components['modules'] = False
        
        # 检查可用的OCR引擎
        if self.ocr_engine:
            components['ocr_engines'] = self.ocr_engine.engines
        
        health_status = "healthy" if all(components.values()) else "degraded"
        
        return jsonify({
            'status': health_status,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime,
            'components': components,
            'request_stats': self.request_stats
        })
    
    def _recognize_plate(self) -> Dict[str, Any]:
        """车牌识别主要API"""
        start_time = time.time()
        
        # 更新活跃请求计数
        if self.metrics:
            self.metrics['active_requests'].inc()
        
        try:
            self.request_stats['total_requests'] += 1
            
            # 获取请求参数
            image_data = self._extract_image_from_request()
            if image_data is None:
                return self._error_response(ErrorCodes.IMAGE_LOAD_ERROR, "无法获取图像数据")
            
            engines = request.form.getlist('engines') or request.json.get('engines') if request.json else None
            enhance = request.form.get('enhance', 'true').lower() == 'true'
            
            # 解码图像
            image = self._decode_image(image_data)
            if image is None:
                return self._error_response(ErrorCodes.IMAGE_LOAD_ERROR, "图像解码失败")
            
            # 执行识别
            result = self._perform_recognition(image, engines, enhance)
            
            # 记录成功
            self.request_stats['successful_requests'] += 1
            processing_time = time.time() - start_time
            
            # 更新监控指标
            if self.metrics:
                self.metrics['requests_total'].labels(method='POST', status='success').inc()
                self.metrics['request_duration'].observe(processing_time)
                if result.get('confidence'):
                    self.metrics['recognition_accuracy'].set(result['confidence'])
            
            result['processing_time'] = processing_time
            result['timestamp'] = datetime.now().isoformat()
            
            return jsonify(result)
            
        except Exception as e:
            self.logger.error(f"识别过程出错: {e}")
            self.request_stats['failed_requests'] += 1
            
            if self.metrics:
                self.metrics['requests_total'].labels(method='POST', status='error').inc()
                self.metrics['errors_total'].labels(error_type='recognition_error').inc()
            
            return self._error_response(ErrorCodes.OCR_ERROR, str(e))
        
        finally:
            # 减少活跃请求计数
            if self.metrics:
                self.metrics['active_requests'].dec()
    
    def _batch_recognize(self) -> Dict[str, Any]:
        """批量识别API"""
        start_time = time.time()
        
        try:
            self.request_stats['total_requests'] += 1
            
            # 获取图像列表
            images_data = []
            
            # 处理多种输入格式
            if 'images' in request.files:
                # 文件上传方式
                for file in request.files.getlist('images'):
                    if file.filename:
                        images_data.append(file.read())
            elif request.json and 'images' in request.json:
                # JSON方式
                for img_b64 in request.json['images']:
                    try:
                        img_data = base64.b64decode(img_b64)
                        images_data.append(img_data)
                    except Exception:
                        continue
            
            if not images_data:
                return self._error_response(ErrorCodes.IMAGE_LOAD_ERROR, "没有有效的图像数据")
            
            # 批量处理
            results = []
            for i, img_data in enumerate(images_data):
                try:
                    image = self._decode_image(img_data)
                    if image is not None:
                        result = self._perform_recognition(image, None, True)
                        result['image_index'] = i
                        results.append(result)
                    else:
                        results.append({
                            'success': False,
                            'image_index': i,
                            'error': '图像解码失败'
                        })
                except Exception as e:
                    results.append({
                        'success': False,
                        'image_index': i,
                        'error': str(e)
                    })
            
            # 统计结果
            successful_count = sum(1 for r in results if r.get('success', False))
            total_count = len(results)
            
            self.request_stats['successful_requests'] += 1
            processing_time = time.time() - start_time
            
            return jsonify({
                'success': True,
                'total_images': total_count,
                'successful_recognitions': successful_count,
                'success_rate': successful_count / total_count if total_count > 0 else 0,
                'results': results,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"批量识别出错: {e}")
            self.request_stats['failed_requests'] += 1
            return self._error_response(ErrorCodes.OCR_ERROR, str(e))
    
    def _analyze_image(self) -> Dict[str, Any]:
        """详细图像分析API"""
        start_time = time.time()
        
        try:
            self.request_stats['total_requests'] += 1
            
            # 获取图像
            image_data = self._extract_image_from_request()
            if image_data is None:
                return self._error_response(ErrorCodes.IMAGE_LOAD_ERROR, "无法获取图像数据")
            
            image = self._decode_image(image_data)
            if image is None:
                return self._error_response(ErrorCodes.IMAGE_LOAD_ERROR, "图像解码失败")
            
            analysis_result = {
                'success': True,
                'image_info': {
                    'shape': image.shape,
                    'dtype': str(image.dtype),
                    'size_bytes': image.nbytes
                },
                'preprocessing': {},
                'detection': {},
                'recognition': {},
                'performance': {}
            }
            
            # 1. 图像预处理分析
            if self.image_processor:
                preprocess_start = time.time()
                processed_results = self.image_processor.preprocess_image(image, method="comprehensive")
                analysis_result['preprocessing'] = {
                    'methods_applied': list(processed_results.keys()),
                    'processing_time': time.time() - preprocess_start
                }
            
            # 2. 车牌检测分析
            if self.plate_detector:
                detection_start = time.time()
                detections = self.plate_detector.detect_plates(image, method="comprehensive")
                analysis_result['detection'] = {
                    'candidates_found': len(detections),
                    'detection_time': time.time() - detection_start,
                    'candidates': []
                }
                
                for i, detection in enumerate(detections[:3]):  # 只返回前3个候选
                    candidate_info = {
                        'index': i,
                        'bbox': detection.get('bbox'),
                        'confidence': detection.get('confidence'),
                        'method': detection.get('method'),
                        'area': detection.get('area'),
                        'aspect_ratio': detection.get('aspect_ratio')
                    }
                    
                    # 验证车牌区域质量
                    if 'region' in detection:
                        validation = self.plate_detector.validate_plate_region(detection['region'])
                        candidate_info['validation'] = validation
                    
                    analysis_result['detection']['candidates'].append(candidate_info)
            
            # 3. OCR识别分析
            if self.ocr_engine:
                recognition_start = time.time()
                
                # 选择最佳检测结果进行识别
                if detections:
                    best_detection = max(detections, key=lambda x: x.get('confidence', 0))
                    recognition_image = best_detection.get('region', image)
                else:
                    recognition_image = image
                
                # 获取所有可用引擎的结果
                ocr_result = self.ocr_engine.recognize(recognition_image, parallel=True)
                
                analysis_result['recognition'] = {
                    'recognition_time': time.time() - recognition_start,
                    'engines_used': ocr_result.get('engines_used', []),
                    'best_result': {
                        'text': ocr_result.get('text', ''),
                        'confidence': ocr_result.get('confidence', 0.0),
                        'engine': ocr_result.get('best_engine', '')
                    },
                    'all_results': ocr_result.get('all_results', {}),
                    'consensus': ocr_result.get('consensus', False)
                }
            
            # 4. 性能分析
            total_time = time.time() - start_time
            analysis_result['performance'] = {
                'total_processing_time': total_time,
                'stages': {
                    'preprocessing': analysis_result.get('preprocessing', {}).get('processing_time', 0),
                    'detection': analysis_result.get('detection', {}).get('detection_time', 0),
                    'recognition': analysis_result.get('recognition', {}).get('recognition_time', 0)
                }
            }
            
            analysis_result['timestamp'] = datetime.now().isoformat()
            self.request_stats['successful_requests'] += 1
            
            return jsonify(analysis_result)
            
        except Exception as e:
            self.logger.error(f"图像分析出错: {e}")
            self.request_stats['failed_requests'] += 1
            return self._error_response(ErrorCodes.OCR_ERROR, str(e))
    
    def _get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        uptime = (datetime.now() - self.request_stats['start_time']).total_seconds()
        
        stats = {
            'service_info': {
                'name': '车牌识别API服务',
                'version': '1.0.0',
                'uptime_seconds': uptime,
                'start_time': self.request_stats['start_time'].isoformat()
            },
            'request_stats': self.request_stats.copy(),
            'performance': {
                'requests_per_second': self.request_stats['total_requests'] / uptime if uptime > 0 else 0,
                'success_rate': self.request_stats['successful_requests'] / self.request_stats['total_requests'] if self.request_stats['total_requests'] > 0 else 0,
                'error_rate': self.request_stats['failed_requests'] / self.request_stats['total_requests'] if self.request_stats['total_requests'] > 0 else 0
            }
        }
        
        # 添加系统资源信息
        try:
            import psutil
            stats['system'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        except ImportError:
            pass
        
        return jsonify(stats)
    
    def _extract_image_from_request(self) -> Optional[bytes]:
        """从请求中提取图像数据"""
        try:
            # 检查文件上传
            if 'image' in request.files:
                file = request.files['image']
                if file.filename:
                    return file.read()
            
            # 检查JSON中的base64数据
            if request.json and 'image' in request.json:
                try:
                    return base64.b64decode(request.json['image'])
                except Exception:
                    pass
            
            # 检查表单中的base64数据
            if 'image' in request.form:
                try:
                    return base64.b64decode(request.form['image'])
                except Exception:
                    pass
            
            return None
            
        except Exception as e:
            self.logger.error(f"提取图像数据失败: {e}")
            return None
    
    def _decode_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """解码图像数据"""
        try:
            # 将字节数据转换为numpy数组
            nparr = np.frombuffer(image_data, np.uint8)
            
            # 使用OpenCV解码
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return image
            
        except Exception as e:
            self.logger.error(f"图像解码失败: {e}")
            return None
    
    def _perform_recognition(self, image: np.ndarray, engines: Optional[list] = None, enhance: bool = True) -> Dict[str, Any]:
        """执行车牌识别"""
        if not MODULES_AVAILABLE:
            return {
                'success': False,
                'error_code': ErrorCodes.MODEL_ERROR,
                'message': '识别模块不可用'
            }
        
        result = {
            'success': False,
            'plate_number': '',
            'confidence': 0.0,
            'engine_used': '',
            'detection_info': {}
        }
        
        try:
            # 1. 图像预处理和增强
            recognition_image = image
            if enhance and self.image_processor:
                processed_results = self.image_processor.preprocess_image(image)
                if 'enhanced' in processed_results:
                    recognition_image = processed_results['enhanced']
            
            # 2. 车牌检测
            detections = []
            if self.plate_detector:
                detections = self.plate_detector.detect_plates(recognition_image)
                result['detection_info'] = {
                    'candidates_found': len(detections),
                    'best_confidence': max((d.get('confidence', 0) for d in detections), default=0)
                }
            
            # 3. 选择最佳检测结果
            if detections:
                best_detection = max(detections, key=lambda x: x.get('confidence', 0))
                plate_region = best_detection.get('region')
                
                # 增强车牌区域
                if plate_region is not None and self.image_processor:
                    plate_region = self.image_processor.enhance_plate_region(plate_region)
                
                ocr_image = plate_region if plate_region is not None else recognition_image
            else:
                ocr_image = recognition_image
            
            # 4. OCR识别
            if self.ocr_engine:
                ocr_result = self.ocr_engine.recognize(ocr_image, engines=engines)
                
                if ocr_result.get('success', False):
                    result['success'] = True
                    result['plate_number'] = ocr_result.get('text', '')
                    result['confidence'] = ocr_result.get('confidence', 0.0)
                    result['engine_used'] = ocr_result.get('best_engine', '')
                    
                    # 添加详细信息
                    if 'all_results' in ocr_result:
                        result['all_engine_results'] = ocr_result['all_results']
                    
                    result['consensus'] = ocr_result.get('consensus', False)
                else:
                    result['error'] = ocr_result.get('message', '识别失败')
            else:
                result['error'] = 'OCR引擎不可用'
            
            return result
            
        except Exception as e:
            self.logger.error(f"识别过程出错: {e}")
            result['error'] = str(e)
            return result
    
    def _error_response(self, error_code: int, message: str) -> Dict[str, Any]:
        """生成错误响应"""
        if self.metrics:
            self.metrics['errors_total'].labels(error_type=f'error_{error_code}').inc()
        
        return jsonify({
            'success': False,
            'error_code': error_code,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }), 400
    
    def run(self, host: str = None, port: int = None, debug: bool = False):
        """启动API服务"""
        if not self.app:
            self.logger.error("Flask应用未初始化")
            return
        
        host = host or self.config.FLASK_HOST
        port = port or self.config.FLASK_PORT
        
        self.logger.info(f"启动车牌识别API服务: http://{host}:{port}")
        self.logger.info(f"API文档: http://{host}:{port}/")
        self.logger.info(f"健康检查: http://{host}:{port}/health")
        
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except Exception as e:
            self.logger.error(f"API服务启动失败: {e}")

def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建并启动API服务
    api = PlateRecognitionAPI()
    api.run(debug=False)

if __name__ == '__main__':
    main()
