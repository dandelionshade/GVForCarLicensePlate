# -*- coding: utf-8 -*-
"""
核心配置模块 - 重构和优化版本

将原来的config.py重构为更加模块化和可扩展的配置系统
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class OCREngineConfig:
    """OCR引擎配置"""
    tesseract_cmd: str = r'E:\application\PDF24\tesseract\tesseract.exe'
    tesseract_config: str = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
    paddle_use_angle_cls: bool = True
    paddle_lang: str = 'ch'
    paddle_use_gpu: bool = False


@dataclass
class ImageProcessingConfig:
    """图像处理配置"""
    max_size: Tuple[int, int] = (1920, 1080)
    plate_min_area: int = 500
    plate_max_area: int = 50000
    plate_aspect_ratio_min: float = 1.5
    plate_aspect_ratio_max: float = 8.0


@dataclass
class ModelConfig:
    """模型配置"""
    input_size: Tuple[int, int] = (120, 32)  # 宽x高
    channels: int = 3
    num_classes: int = 68
    model_path: str = "models/"
    device: str = "cpu"  # cpu 或 cuda


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ['*'])
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    upload_folder: str = "web/static/uploads"


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/plate_recognition.log"
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class Config:
    """主配置类 - 统一管理所有配置"""
    
    def __init__(self, config_file: Optional[str] = None, environment: str = "development"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent.parent
        
        # 基础配置
        self.debug = environment == "development"
        self.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
        
        # API密钥 (建议从环境变量读取)
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY', 'AIzaSyARL4h588FeWT-eSUdQPqfZeWcmifLDjb0')
        
        # 各模块配置
        self.ocr = OCREngineConfig()
        self.image_processing = ImageProcessingConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        
        # 车牌相关常量
        self.provinces = [
            '京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘', '皖', 
            '鲁', '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕', 
            '吉', '闽', '贵', '粤', '青', '藏', '川', '宁', '琼'
        ]
        
        self.plate_pattern = r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][0-9A-Z]{4,6}$'
        
        # 如果提供了配置文件，加载外部配置
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """从配置文件加载配置"""
        config_path = Path(config_file)
        if not config_path.exists():
            config_path = self.project_root / "configs" / f"{config_file}.yaml"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                self._update_from_dict(config_data)
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (OCREngineConfig, ImageProcessingConfig, 
                                                 ModelConfig, APIConfig, LoggingConfig)):
                    # 更新嵌套配置对象
                    config_obj = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(config_obj, sub_key):
                            setattr(config_obj, sub_key, sub_value)
                else:
                    setattr(self, key, value)
    
    def get_model_path(self, model_name: str) -> Path:
        """获取模型文件路径"""
        return self.project_root / self.model.model_path / model_name
    
    def get_log_path(self) -> Path:
        """获取日志文件路径"""
        log_path = self.project_root / self.logging.file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path
    
    def get_upload_path(self) -> Path:
        """获取上传文件路径"""
        upload_path = self.project_root / self.api.upload_folder
        upload_path.mkdir(parents=True, exist_ok=True)
        return upload_path
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'ocr': self.ocr.__dict__,
            'image_processing': self.image_processing.__dict__,
            'model': self.model.__dict__,
            'api': self.api.__dict__,
            'logging': self.logging.__dict__,
            'provinces': self.provinces,
            'plate_pattern': self.plate_pattern
        }


# 全局配置实例
_config_instance = None

def get_config(environment: str = None, config_file: str = None) -> Config:
    """获取配置实例（单例模式）"""
    global _config_instance
    
    if _config_instance is None:
        env = environment or os.environ.get('APP_ENV', 'development')
        _config_instance = Config(config_file=config_file, environment=env)
    
    return _config_instance


def reload_config(environment: str = None, config_file: str = None):
    """重新加载配置"""
    global _config_instance
    _config_instance = None
    return get_config(environment, config_file)
