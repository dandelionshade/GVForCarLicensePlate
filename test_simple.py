#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版车牌识别系统测试脚本
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 导入简化的识别器
try:
    from simple_app import SimplePlateRecognizer
    print("✓ 成功导入SimplePlateRecognizer")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)


def create_test_image():
    """创建一个简单的测试图像"""
    # 创建白色背景
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # 绘制蓝色矩形模拟车牌
    cv2.rectangle(img, (50, 50), (350, 150), (255, 100, 0), -1)  # 蓝色背景
    
    # 添加白色文字模拟车牌号
    cv2.putText(img, "京A12345", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    return img


def test_recognizer():
    """测试识别器功能"""
    print("\n=== 车牌识别系统测试 ===\n")
    
    # 初始化识别器
    try:
        recognizer = SimplePlateRecognizer()
        print(f"✓ 识别器初始化成功")
        print(f"  可用OCR引擎: {recognizer.engines}")
    except Exception as e:
        print(f"✗ 识别器初始化失败: {e}")
        return False
    
    # 创建测试图像
    print("\n1. 创建测试图像...")
    test_img = create_test_image()
    
    # 保存测试图像
    test_img_path = "test_plate.jpg"
    cv2.imwrite(test_img_path, test_img)
    print(f"✓ 测试图像已保存: {test_img_path}")
    
    # 测试车牌检测
    print("\n2. 测试车牌区域检测...")
    try:
        regions = recognizer.detect_plate_region(test_img)
        if regions:
            print(f"✓ 检测到 {len(regions)} 个候选区域")
            for i, (x, y, w, h) in enumerate(regions):
                print(f"  区域{i+1}: ({x}, {y}, {w}, {h})")
        else:
            print("⚠ 未检测到车牌区域")
    except Exception as e:
        print(f"✗ 车牌检测失败: {e}")
    
    # 测试完整识别流程
    print("\n3. 测试完整识别流程...")
    try:
        result = recognizer.recognize(test_img)
        
        if result['success']:
            print("✓ 识别成功!")
            print(f"  车牌号: {result['plate_number']}")
            print(f"  置信度: {result['confidence']:.2f}")
            if 'bbox' in result:
                print(f"  位置: {result['bbox']}")
        else:
            print(f"⚠ 识别失败: {result['error']}")
    except Exception as e:
        print(f"✗ 识别过程出错: {e}")
    
    # 测试文本验证
    print("\n4. 测试车牌格式验证...")
    test_plates = [
        "京A12345",   # 有效
        "沪B67890",   # 有效
        "ABC123",     # 无效
        "京AA1234",   # 有效
        "123456",     # 无效
    ]
    
    for plate in test_plates:
        is_valid = recognizer.validate_plate_text(plate)
        status = "✓" if is_valid else "✗"
        print(f"  {status} {plate}: {'有效' if is_valid else '无效'}")
    
    # 清理测试文件
    try:
        os.remove(test_img_path)
        print(f"\n✓ 清理测试文件: {test_img_path}")
    except:
        pass
    
    print("\n=== 测试完成 ===")
    return True


def test_dependencies():
    """测试依赖包"""
    print("=== 依赖包测试 ===\n")
    
    dependencies = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('flask', 'Flask'),
    ]
    
    optional_deps = [
        ('pytesseract', 'Tesseract OCR'),
        ('paddleocr', 'PaddleOCR'),
    ]
    
    # 测试必需依赖
    print("必需依赖:")
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - 未安装")
    
    # 测试可选依赖
    print("\n可选依赖:")
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"⚠ {name} - 未安装（可选）")


def main():
    """主函数"""
    print("车牌识别系统 - 简化版测试工具\n")
    
    # 测试依赖
    test_dependencies()
    
    print()
    
    # 测试识别器
    test_recognizer()
    
    print("\n提示:")
    print("- 如果OCR引擎不可用，请安装对应的依赖包")
    print("- Tesseract需要单独安装程序，不仅仅是Python包")
    print("- PaddleOCR首次使用会下载模型文件")


if __name__ == '__main__':
    main()
