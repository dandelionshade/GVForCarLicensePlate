# -*- coding: utf-8 -*-
"""
API v1版本 - 车牌识别端点

提供RESTful API接口
"""

import time
import base64
import logging
from io import BytesIO
from typing import Optional, List

import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from PIL import Image

from plate_recognition.pipeline.recognition_pipeline import RecognitionPipeline
from plate_recognition.core.exceptions import PlateRecognitionError, ErrorCodes
from plate_recognition.core.constants import PlateConstants


# 创建蓝图
api_v1_bp = Blueprint('api_v1', __name__)

# 初始化识别流水线
pipeline = None

def get_pipeline():
    """获取识别流水线实例（懒加载）"""
    global pipeline
    if pipeline is None:
        pipeline = RecognitionPipeline()
    return pipeline


@api_v1_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        p = get_pipeline()
        stats = p.get_stats()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'components': stats['component_status'],
            'version': '2.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 500


@api_v1_bp.route('/recognize', methods=['POST'])
def recognize_plate():
    """车牌识别主接口"""
    try:
        # 获取图像数据
        image_data = extract_image_from_request()
        if image_data is None:
            raise PlateRecognitionError(
                "无法获取图像数据", 
                ErrorCodes.IMAGE_LOAD_ERROR
            )
        
        # 解码图像
        image = decode_image(image_data)
        if image is None:
            raise PlateRecognitionError(
                "图像解码失败", 
                ErrorCodes.IMAGE_LOAD_ERROR
            )
        
        # 获取参数
        detection_method = request.form.get('detection_method', 'comprehensive')
        ocr_engines = request.form.getlist('ocr_engines')
        enable_preprocessing = request.form.get('enable_preprocessing', 'true').lower() == 'true'
        return_debug_info = request.form.get('debug', 'false').lower() == 'true'
        
        # 执行识别
        p = get_pipeline()
        result = p.recognize(
            image=image,
            detection_method=detection_method,
            ocr_engines=ocr_engines if ocr_engines else None,
            enable_preprocessing=enable_preprocessing,
            return_debug_info=return_debug_info
        )
        
        # 添加时间戳
        result['timestamp'] = time.time()
        
        return jsonify(result)
        
    except PlateRecognitionError:
        raise
    except Exception as e:
        logging.error(f"识别过程错误: {e}")
        raise PlateRecognitionError(
            f"识别过程失败: {e}", 
            ErrorCodes.UNKNOWN_ERROR
        )


@api_v1_bp.route('/recognize/batch', methods=['POST'])
def batch_recognize():
    """批量识别接口"""
    try:
        # 检查是否为JSON请求
        if not request.is_json:
            raise PlateRecognitionError(
                "批量识别需要JSON格式的请求", 
                ErrorCodes.REQUEST_ERROR
            )
        
        data = request.get_json()
        if not data or 'images' not in data:
            raise PlateRecognitionError(
                "缺少images字段", 
                ErrorCodes.REQUEST_ERROR
            )
        
        images_data = data['images']
        if not isinstance(images_data, list):
            raise PlateRecognitionError(
                "images必须是数组", 
                ErrorCodes.REQUEST_ERROR
            )
        
        # 获取参数
        detection_method = data.get('detection_method', 'comprehensive')
        ocr_engines = data.get('ocr_engines')
        enable_preprocessing = data.get('enable_preprocessing', True)
        
        results = []
        p = get_pipeline()
        
        for i, image_data in enumerate(images_data):
            try:
                # 解码base64图像
                if isinstance(image_data, str):
                    image_bytes = base64.b64decode(image_data)
                    image = decode_image(image_bytes)
                else:
                    raise PlateRecognitionError(
                        f"第{i+1}张图像数据格式错误", 
                        ErrorCodes.IMAGE_LOAD_ERROR
                    )
                
                # 执行识别
                result = p.recognize(
                    image=image,
                    detection_method=detection_method,
                    ocr_engines=ocr_engines,
                    enable_preprocessing=enable_preprocessing,
                    return_debug_info=False
                )
                
                result['index'] = i
                results.append(result)
                
            except Exception as e:
                logging.error(f"第{i+1}张图像处理失败: {e}")
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e),
                    'plate_number': '',
                    'confidence': 0.0
                })
        
        return jsonify({
            'success': True,
            'total_processed': len(results),
            'successful_count': sum(1 for r in results if r.get('success')),
            'results': results,
            'timestamp': time.time()
        })
        
    except PlateRecognitionError:
        raise
    except Exception as e:
        logging.error(f"批量识别错误: {e}")
        raise PlateRecognitionError(
            f"批量识别失败: {e}", 
            ErrorCodes.UNKNOWN_ERROR
        )


@api_v1_bp.route('/analyze', methods=['POST'])
def analyze_image():
    """详细图像分析接口"""
    try:
        # 获取图像数据
        image_data = extract_image_from_request()
        if image_data is None:
            raise PlateRecognitionError(
                "无法获取图像数据", 
                ErrorCodes.IMAGE_LOAD_ERROR
            )
        
        # 解码图像
        image = decode_image(image_data)
        if image is None:
            raise PlateRecognitionError(
                "图像解码失败", 
                ErrorCodes.IMAGE_LOAD_ERROR
            )
        
        # 执行详细分析
        p = get_pipeline()
        result = p.recognize(
            image=image,
            detection_method="comprehensive",
            enable_preprocessing=True,
            return_debug_info=True
        )
        
        # 添加图像质量分析
        if hasattr(p, 'image_processor') and p.image_processor:
            quality_analysis = p.image_processor.analyze_image_quality(image)
            result['image_quality'] = quality_analysis
        
        result['timestamp'] = time.time()
        
        return jsonify(result)
        
    except PlateRecognitionError:
        raise
    except Exception as e:
        logging.error(f"图像分析错误: {e}")
        raise PlateRecognitionError(
            f"图像分析失败: {e}", 
            ErrorCodes.UNKNOWN_ERROR
        )


@api_v1_bp.route('/stats', methods=['GET'])
def get_stats():
    """获取服务统计信息"""
    try:
        p = get_pipeline()
        stats = p.get_stats()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logging.error(f"获取统计信息错误: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500


@api_v1_bp.route('/reset-stats', methods=['POST'])
def reset_stats():
    """重置统计信息"""
    try:
        p = get_pipeline()
        p.reset_stats()
        
        return jsonify({
            'success': True,
            'message': '统计信息已重置',
            'timestamp': time.time()
        })
        
    except Exception as e:
        logging.error(f"重置统计信息错误: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500


@api_v1_bp.route('/engines', methods=['GET'])
def get_available_engines():
    """获取可用的OCR引擎"""
    try:
        p = get_pipeline()
        
        if hasattr(p, 'ocr_engine') and p.ocr_engine:
            engines = p.ocr_engine.get_available_engines()
        else:
            engines = []
        
        return jsonify({
            'success': True,
            'engines': engines,
            'count': len(engines),
            'timestamp': time.time()
        })
        
    except Exception as e:
        logging.error(f"获取引擎信息错误: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500


@api_v1_bp.route('/validate', methods=['POST'])
def validate_plate_number():
    """验证车牌号码格式"""
    try:
        if request.is_json:
            data = request.get_json()
            plate_number = data.get('plate_number', '')
        else:
            plate_number = request.form.get('plate_number', '')
        
        if not plate_number:
            raise PlateRecognitionError(
                "缺少plate_number参数", 
                ErrorCodes.VALIDATION_ERROR
            )
        
        # 验证格式
        is_valid = PlateConstants.validate_plate_number(plate_number)
        province = PlateConstants.get_province_from_plate(plate_number)
        city_code = PlateConstants.get_city_code_from_plate(plate_number)
        
        return jsonify({
            'success': True,
            'plate_number': plate_number,
            'is_valid': is_valid,
            'province': province,
            'city_code': city_code,
            'length': len(plate_number),
            'timestamp': time.time()
        })
        
    except PlateRecognitionError:
        raise
    except Exception as e:
        logging.error(f"验证车牌号码错误: {e}")
        raise PlateRecognitionError(
            f"验证失败: {e}", 
            ErrorCodes.VALIDATION_ERROR
        )


def extract_image_from_request() -> Optional[bytes]:
    """从请求中提取图像数据"""
    try:
        # 检查文件上传
        if 'image' in request.files:
            file = request.files['image']
            if file.filename:
                return file.read()
        
        # 检查JSON中的base64数据
        if request.is_json:
            data = request.get_json()
            if data and 'image' in data:
                try:
                    return base64.b64decode(data['image'])
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
        logging.error(f"提取图像数据失败: {e}")
        return None


def decode_image(image_data: bytes) -> Optional[np.ndarray]:
    """解码图像数据"""
    try:
        # 方法1: 使用OpenCV直接解码
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is not None:
            return image
        
        # 方法2: 使用PIL解码
        try:
            pil_image = Image.open(BytesIO(image_data))
            # 转换为RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # 转换为OpenCV格式（BGR）
            image_array = np.array(pil_image)
            image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            return image
        except Exception:
            pass
        
        return None
        
    except Exception as e:
        logging.error(f"图像解码失败: {e}")
        return None
