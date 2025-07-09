#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量车牌识别工具 - 增强版
支持处理指定目录中的所有图片
"""

import os
import sys
import argparse
import cv2
import json
from pathlib import Path
from datetime import datetime
import time

# 导入简化的识别器
try:
    from simple_app import SimplePlateRecognizer, ALLOWED_EXTENSIONS
    print("✓ 成功导入SimplePlateRecognizer")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("请确保 simple_app.py 在同一目录下")
    sys.exit(1)


def get_image_files(directory):
    """获取目录中的所有图片文件"""
    image_files = []
    directory = Path(directory)
    
    if not directory.exists():
        print(f"错误: 目录不存在 - {directory}")
        return []
    
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower()[1:] in ALLOWED_EXTENSIONS:
            image_files.append(file_path)
    
    return sorted(image_files)


def process_single_image(recognizer, image_path, save_results=False, output_dir=None):
    """处理单张图片"""
    try:
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                'file': str(image_path),
                'success': False,
                'error': '无法读取图像文件'
            }
        
        # 执行识别
        start_time = time.time()
        result = recognizer.recognize(image)
        processing_time = time.time() - start_time
        
        # 添加文件信息和处理时间
        result['file'] = str(image_path.name)
        result['file_path'] = str(image_path)
        result['processing_time'] = round(processing_time, 3)
        result['timestamp'] = datetime.now().isoformat()
        
        # 保存结果到文件
        if save_results and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_filename = f"{timestamp}_{image_path.stem}_result.json"
            result_path = output_dir / result_filename
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            result['result_file'] = str(result_path)
        
        return result
        
    except Exception as e:
        return {
            'file': str(image_path.name),
            'file_path': str(image_path),
            'success': False,
            'error': f'处理异常: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }


def batch_process(input_dir, output_dir=None, save_individual=False, verbose=False):
    """批量处理图片"""
    
    print(f"开始批量处理: {input_dir}")
    
    # 获取图片文件列表
    image_files = get_image_files(input_dir)
    
    if not image_files:
        print("未找到图片文件")
        return []
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 初始化识别器
    recognizer = SimplePlateRecognizer()
    print(f"可用OCR引擎: {recognizer.engines}")
    
    if not recognizer.engines:
        print("警告: 没有可用的OCR引擎，识别可能失败")
    
    # 处理所有图片
    results = []
    success_count = 0
    start_time = time.time()
    
    for i, image_path in enumerate(image_files, 1):
        if verbose:
            print(f"\n[{i}/{len(image_files)}] 处理: {image_path.name}")
        else:
            print(f"进度: {i}/{len(image_files)}", end='\r')
        
        # 处理单张图片
        result = process_single_image(
            recognizer, 
            image_path, 
            save_individual, 
            output_dir
        )
        
        results.append(result)
        
        if result['success']:
            success_count += 1
            if verbose:
                print(f"  ✓ 识别成功: {result['plate_number']} (置信度: {result['confidence']:.2f})")
        else:
            if verbose:
                print(f"  ✗ 识别失败: {result['error']}")
    
    total_time = time.time() - start_time
    
    # 统计结果
    print(f"\n\n=== 批量处理完成 ===")
    print(f"总数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(results) - success_count}")
    print(f"成功率: {success_count / len(results) * 100:.1f}%")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均耗时: {total_time / len(results):.2f}秒/张")
    
    # 保存汇总结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'input_directory': str(input_dir),
            'total_images': len(results),
            'successful': success_count,
            'failed': len(results) - success_count,
            'success_rate': success_count / len(results) * 100,
            'total_time': total_time,
            'average_time': total_time / len(results),
            'available_engines': recognizer.engines,
            'results': results
        }
        
        summary_file = output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"汇总结果已保存到: {summary_file}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量车牌识别工具')
    parser.add_argument('input_dir', help='输入图片目录')
    parser.add_argument('--output', '-o', help='输出结果目录')
    parser.add_argument('--save-individual', '-s', action='store_true', 
                       help='保存每张图片的识别结果')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在 - {args.input_dir}")
        sys.exit(1)
    
    # 设置输出目录
    output_dir = args.output or "data/results"
    
    try:
        # 执行批量处理
        results = batch_process(
            args.input_dir,
            output_dir,
            args.save_individual,
            args.verbose
        )
        
        # 显示成功的识别结果
        if not args.verbose:
            successful_results = [r for r in results if r['success']]
            if successful_results:
                print("\n成功识别的车牌:")
                for result in successful_results:
                    print(f"  {result['file']}: {result['plate_number']} "
                          f"(置信度: {result['confidence']:.2f})")
        
    except KeyboardInterrupt:
        print("\n\n用户中断了批量处理")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n批量处理出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
