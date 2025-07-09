#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
车牌识别命令行工具 - 简化版本
"""

import os
import sys
import argparse
import cv2
import json
from pathlib import Path

# 导入简化的识别器
from simple_app import SimplePlateRecognizer


def main():
    parser = argparse.ArgumentParser(description='车牌识别命令行工具')
    parser.add_argument('image_path', help='图像文件路径')
    parser.add_argument('--output', '-o', help='输出结果到JSON文件')
    parser.add_argument('--show', '-s', action='store_true', help='显示检测结果图像')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.image_path):
        print(f"错误: 文件不存在 - {args.image_path}")
        sys.exit(1)
    
    # 读取图像
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"错误: 无法读取图像文件 - {args.image_path}")
        sys.exit(1)
    
    # 初始化识别器
    recognizer = SimplePlateRecognizer()
    
    print(f"使用OCR引擎: {recognizer.engines}")
    print(f"处理图像: {args.image_path}")
    
    # 执行识别
    result = recognizer.recognize(image)
    
    # 输出结果
    if result['success']:
        print(f"✓ 识别成功!")
        print(f"  车牌号: {result['plate_number']}")
        print(f"  置信度: {result['confidence']:.2f}")
        if 'bbox' in result:
            x, y, w, h = result['bbox']
            print(f"  位置: ({x}, {y}, {w}, {h})")
    else:
        print(f"✗ 识别失败: {result['error']}")
    
    # 保存结果到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {args.output}")
    
    # 显示图像
    if args.show and result['success'] and 'bbox' in result:
        x, y, w, h = result['bbox']
        # 在图像上绘制边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 在边界框上方显示识别结果
        cv2.putText(image, result['plate_number'], (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('车牌识别结果', image)
        print("按任意键关闭图像窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
