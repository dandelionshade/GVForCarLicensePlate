# -*- coding: utf-8 -*-
"""
自定义异常类 - 车牌识别系统

定义系统中使用的各种异常类型，便于错误处理和调试
"""

class PlateRecognitionError(Exception):
    """车牌识别系统基础异常类"""
    
    def __init__(self, message: str, error_code: int = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(PlateRecognitionError):
    """配置错误"""
    pass


class ModelLoadError(PlateRecognitionError):
    """模型加载错误"""
    pass


class ImageProcessingError(PlateRecognitionError):
    """图像处理错误"""
    pass


class DetectionError(PlateRecognitionError):
    """车牌检测错误"""
    pass


class RecognitionError(PlateRecognitionError):
    """OCR识别错误"""
    pass


class ValidationError(PlateRecognitionError):
    """数据验证错误"""
    pass


class APIError(PlateRecognitionError):
    """API错误"""
    pass


# 错误码常量
class ErrorCodes:
    """错误码定义"""
    
    # 通用错误 (1000-1999)
    UNKNOWN_ERROR = 1000
    CONFIGURATION_ERROR = 1001
    VALIDATION_ERROR = 1002
    
    # 图像处理错误 (2000-2999)
    IMAGE_LOAD_ERROR = 2000
    IMAGE_FORMAT_ERROR = 2001
    IMAGE_SIZE_ERROR = 2002
    IMAGE_PROCESSING_ERROR = 2003
    
    # 模型错误 (3000-3999)
    MODEL_LOAD_ERROR = 3000
    MODEL_PREDICTION_ERROR = 3001
    MODEL_NOT_FOUND = 3002
    
    # 检测错误 (4000-4999)
    DETECTION_ERROR = 4000
    NO_PLATE_DETECTED = 4001
    DETECTION_CONFIDENCE_LOW = 4002
    
    # 识别错误 (5000-5999)
    OCR_ERROR = 5000
    OCR_ENGINE_ERROR = 5001
    RECOGNITION_FAILED = 5002
    
    # API错误 (6000-6999)
    API_ERROR = 6000
    REQUEST_ERROR = 6001
    RESPONSE_ERROR = 6002
    AUTHENTICATION_ERROR = 6003
    
    # 系统错误 (7000-7999)
    SYSTEM_ERROR = 7000
    RESOURCE_ERROR = 7001
    PERMISSION_ERROR = 7002
