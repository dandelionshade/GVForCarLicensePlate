# -*- coding: utf-8 -*-
"""
Flask API应用工厂

重构后的API服务，更加模块化和可维护
"""

import os
import logging
from flask import Flask

# 条件导入CORS
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Warning: flask-cors not available, CORS support disabled")

from plate_recognition.core.config import get_config
from plate_recognition.core.exceptions import PlateRecognitionError, ErrorCodes


def create_app(config_override=None):
    """创建API应用实例"""
    
    # 获取配置
    config = get_config()
    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # 创建Flask应用
    app = Flask(__name__)
    
    # 配置应用
    app.config.update({
        'SECRET_KEY': config.secret_key,
        'DEBUG': config.debug,
        'JSON_AS_ASCII': False,
        'MAX_CONTENT_LENGTH': config.api.max_content_length,
    })
    
    # 启用CORS（如果可用）
    if CORS_AVAILABLE:
        cors = CORS(app, origins=config.api.cors_origins)
    else:
        # 手动添加CORS头
        @app.after_request
        def after_request(response):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            return response
    
    # 设置日志
    if not app.debug:
        logging.basicConfig(
            level=getattr(logging, config.logging.level),
            format=config.logging.format
        )
    
    # 注册蓝图
    from .v1 import api_v1_bp
    app.register_blueprint(api_v1_bp, url_prefix='/api/v1')
    
    # 注册错误处理器
    register_error_handlers(app)
    
    return app


def register_error_handlers(app):
    """注册错误处理器"""
    
    @app.errorhandler(PlateRecognitionError)
    def handle_recognition_error(error):
        return {
            'success': False,
            'error': error.message,
            'error_code': error.error_code
        }, 400
    
    @app.errorhandler(404)
    def not_found(error):
        return {
            'success': False,
            'error': 'API endpoint not found',
            'error_code': ErrorCodes.API_ERROR
        }, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {
            'success': False,
            'error': 'Internal server error',
            'error_code': ErrorCodes.SYSTEM_ERROR
        }, 500
    
    @app.errorhandler(413)
    def too_large(error):
        return {
            'success': False,
            'error': 'File too large',
            'error_code': ErrorCodes.REQUEST_ERROR
        }, 413
    
    @app.errorhandler(400)
    def bad_request(error):
        return {
            'success': False,
            'error': 'Bad request',
            'error_code': ErrorCodes.REQUEST_ERROR
        }, 400

    return app
