"""
车牌识别系统运行入口
整合所有功能，提供命令行和GUI两种运行方式
"""
import sys
import os
import argparse
import cv2
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging(level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('system_runtime.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def run_gui_mode():
    """运行GUI模式（原有的Tkinter界面）"""
    try:
        print("启动GUI模式...")
        import tkinter as tk
        from main import HybridLPR_App
        
        root = tk.Tk()
        app = HybridLPR_App(root)
        root.mainloop()
        
    except ImportError as e:
        print(f"GUI模式启动失败: {e}")
        print("请确保已安装所需的依赖包")
    except Exception as e:
        print(f"GUI运行错误: {e}")

def run_cli_mode(image_path: str, method: str = "fusion"):
    """运行命令行模式"""
    try:
        print(f"启动命令行模式，处理图像: {image_path}")
        
        # 导入优化系统
        from optimized_system import OptimizedPlateRecognitionSystem
        
        # 初始化系统
        system = OptimizedPlateRecognitionSystem()
        
        # 加载图像
        if not os.path.exists(image_path):
            print(f"错误: 图像文件不存在 - {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法加载图像 - {image_path}")
            return
        
        print(f"图像加载成功，尺寸: {image.shape}")
        
        # 执行识别
        print(f"开始识别，使用方法: {method}")
        result = system.recognize_plate(image, method)
        
        # 显示结果
        print("\\n=== 识别结果 ===")
        print(f"识别成功: {result.get('success', False)}")
        print(f"车牌号码: {result.get('plate_text', '未识别')}")
        print(f"置信度: {result.get('confidence', 0.0):.2f}")
        print(f"处理时间: {result.get('processing_time', 0.0):.3f}秒")
        print(f"识别方法: {result.get('method', 'unknown')}")
        
        if 'details' in result:
            print(f"详细信息: {result['details']}")
        
        # 性能报告
        performance = system.get_performance_report()
        print("\\n=== 性能统计 ===")
        print(f"总处理数: {performance['metrics']['total_processed']}")
        print(f"成功率: {performance['metrics']['accuracy_rate']:.1f}%")
        print(f"平均耗时: {performance['metrics']['average_time']:.3f}秒")
        
    except ImportError as e:
        print(f"命令行模式启动失败: {e}")
        print("某些高级功能不可用，尝试基础识别...")
        run_basic_cli(image_path)
    except Exception as e:
        print(f"命令行模式运行错误: {e}")

def run_basic_cli(image_path: str):
    """运行基础命令行识别"""
    try:
        import pytesseract
        import cv2
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法加载图像 - {image_path}")
            return
        
        # 基础预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR识别
        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领'
        text = pytesseract.image_to_string(binary, config=config).strip()
        text = ''.join(c for c in text if c.isalnum() or c in '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领')
        
        print("\\n=== 基础识别结果 ===")
        print(f"识别结果: {text if text else '未识别到车牌'}")
        
    except Exception as e:
        print(f"基础识别失败: {e}")

def run_batch_mode(input_dir: str, output_file: str = "batch_results.json"):
    """运行批量处理模式"""
    try:
        print(f"启动批量处理模式，输入目录: {input_dir}")
        
        from optimized_system import OptimizedPlateRecognitionSystem
        import json
        
        # 获取所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for file_path in Path(input_dir).iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path))
        
        if not image_paths:
            print(f"在 {input_dir} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_paths)} 个图像文件")
        
        # 初始化系统
        system = OptimizedPlateRecognitionSystem()
        
        # 批量处理
        print("开始批量处理...")
        results = system.batch_process(image_paths, method="fusion")
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 统计结果
        success_count = sum(1 for r in results if r.get('success', False))
        success_rate = success_count / len(results) * 100
        
        print(f"\\n=== 批量处理完成 ===")
        print(f"总文件数: {len(results)}")
        print(f"成功识别: {success_count}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"结果已保存至: {output_file}")
        
        # 显示前几个结果
        print("\\n=== 部分结果预览 ===")
        for i, result in enumerate(results[:5]):
            filename = Path(result.get('image_path', '')).name
            plate_text = result.get('plate_text', '未识别')
            print(f"{i+1}. {filename}: {plate_text}")
        
    except Exception as e:
        print(f"批量处理失败: {e}")

def run_api_mode(host: str = "0.0.0.0", port: int = 5000):
    """运行API服务模式"""
    try:
        print(f"启动API服务模式，地址: http://{host}:{port}")
        
        from api_server import FlaskAPIServer
        
        # 创建API服务器
        api_server = FlaskAPIServer()
        
        print("API服务启动中...")
        print(f"访问地址: http://{host}:{port}")
        print("API文档: http://{host}:{port}/docs")
        print("健康检查: http://{host}:{port}/health")
        
        # 启动服务
        api_server.run(host=host, port=port)
        
    except ImportError as e:
        print(f"API服务启动失败: {e}")
        print("请确保已安装Flask相关依赖")
    except Exception as e:
        print(f"API服务运行错误: {e}")

def run_test_mode():
    """运行测试模式"""
    try:
        print("启动系统测试模式...")
        
        from optimized_system import OptimizedPlateRecognitionSystem
        from test_framework import PlateRecognitionTestSuite
        
        # 系统基础测试
        print("1. 系统基础测试...")
        system = OptimizedPlateRecognitionSystem()
        basic_test = system.run_system_test()
        
        print(f"   组件状态: {basic_test.get('components', {})}")
        print(f"   基础测试: {basic_test.get('basic_test', {}).get('success', False)}")
        
        # 全面测试（如果测试框架可用）
        try:
            print("2. 全面系统测试...")
            test_suite = PlateRecognitionTestSuite()
            comprehensive_test = test_suite.run_comprehensive_test()
            
            print(f"   测试完成时间: {comprehensive_test.get('timestamp', 'N/A')}")
            print(f"   测试结果摘要: {comprehensive_test.get('summary', {})}")
            
        except Exception as e:
            print(f"   全面测试跳过: {e}")
        
        print("\\n系统测试完成")
        
    except Exception as e:
        print(f"测试模式失败: {e}")

def main():
    """主入口函数"""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="车牌识别系统运行工具")
    parser.add_argument('--mode', choices=['gui', 'cli', 'batch', 'api', 'test'], 
                       default='gui', help='运行模式')
    parser.add_argument('--image', type=str, help='图像文件路径（CLI模式）')
    parser.add_argument('--input-dir', type=str, help='输入目录（批量模式）')
    parser.add_argument('--output', type=str, default='batch_results.json', 
                       help='输出文件（批量模式）')
    parser.add_argument('--method', choices=['basic', 'advanced', 'fusion'], 
                       default='fusion', help='识别方法')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='API服务主机')
    parser.add_argument('--port', type=int, default=5000, help='API服务端口')
    
    args = parser.parse_args()
    
    print("=== 车牌识别系统 ===")
    print(f"运行模式: {args.mode}")
    print("-" * 30)
    
    if args.mode == 'gui':
        run_gui_mode()
    elif args.mode == 'cli':
        if not args.image:
            print("错误: CLI模式需要指定 --image 参数")
            sys.exit(1)
        run_cli_mode(args.image, args.method)
    elif args.mode == 'batch':
        if not args.input_dir:
            print("错误: 批量模式需要指定 --input-dir 参数")
            sys.exit(1)
        run_batch_mode(args.input_dir, args.output)
    elif args.mode == 'api':
        run_api_mode(args.host, args.port)
    elif args.mode == 'test':
        run_test_mode()

if __name__ == "__main__":
    main()
