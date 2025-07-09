# -*- coding: utf-8 -*-
"""
配置加载器
"""

import os
import importlib

def load_config(env=None):
    """加载配置"""
    if env is None:
        env = os.getenv('ENVIRONMENT', 'development')
    
    try:
        config_module = importlib.import_module(f'configs.{env}')
        return config_module
    except ImportError:
        # 默认使用开发配置
        config_module = importlib.import_module('configs.development')
        return config_module

# 全局配置实例
config = load_config()
