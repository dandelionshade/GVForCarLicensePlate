#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成测试车牌图片
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_sample_plate_image(plate_text, filename, save_dir):
    """创建示例车牌图片"""
    # 创建车牌背景 (蓝色)
    img = np.ones((140, 440, 3), dtype=np.uint8)
    img[:, :] = [255, 100, 0]  # 蓝色背景
    
    # 添加白色边框
    cv2.rectangle(img, (5, 5), (435, 135), (255, 255, 255), 2)
    
    # 添加车牌文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)  # 白色文字
    
    # 省份文字 (较大)
    cv2.putText(img, plate_text[0], (20, 85), font, 2.5, text_color, 4)
    
    # 其余文字
    x_pos = 80
    for char in plate_text[1:]:
        cv2.putText(img, char, (x_pos, 85), font, 2.2, text_color, 3)
        x_pos += 45
    
    # 保存图片
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, img)
    print(f"创建测试图片: {save_path}")

def main():
    """创建多个测试车牌图片"""
    project_root = Path(__file__).parent
    test_dir = project_root / "data" / "test_images"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成不同的车牌样本
    test_plates = [
        ("京A12345", "jing_a12345.jpg"),
        ("沪B67890", "hu_b67890.jpg"),
        ("粤C88888", "yue_c88888.jpg"),
        ("川D99999", "chuan_d99999.jpg"),
        ("浙E00001", "zhe_e00001.jpg"),
        ("苏F77777", "su_f77777.jpg"),
    ]
    
    print("正在生成测试车牌图片...")
    for plate_text, filename in test_plates:
        create_sample_plate_image(plate_text, filename, str(test_dir))
    
    print(f"\n完成！共生成 {len(test_plates)} 张测试图片")
    print(f"图片保存在: {test_dir}")

if __name__ == "__main__":
    main()
