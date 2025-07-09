'''
Author: zhen doniajohary2677@gmail.com
Date: 2025-07-09 10:46:12
LastEditors: zhen doniajohary2677@gmail.com
LastEditTime: 2025-07-09 13:50:42
FilePath: \\GVForCarLicensePlate\\launcher.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
车牌识别系统 - 统一入口点

整合所有运行方式：Web服务、CLI命令行、GUI界面
支持多种部署模式和用法
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 添加项目路径到Python搜索路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging(debug=False):
    """设置日志配置"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('plate_recognition_system.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def run_web_server(host='0.0.0.0', port=5000, debug=False):
    """运行Web服务器"""
    # 检查Web依赖
    web_deps = [
        ('flask', 'flask'),
        ('flask-cors', 'flask_cors')
    ]
    missing_deps = check_dependencies(web_deps)
    if missing_deps:
        print("❌ Web服务器启动失败")
        print_dependency_help(missing_deps, "Web服务器")
        return 1
    
    try:
        from api.app import create_app
        app = create_app()
        
        print(f"🚀 启动Web服务器...")
        print(f"📍 访问地址: http://{host}:{port}")
        print(f"📖 API文档: http://{host}:{port}/api/docs")
        print(f"🎯 在线识别: http://{host}:{port}/upload")
        
        app.run(host=host, port=port, debug=debug, threaded=True)
        return 0
        
    except ImportError as e:
        print(f"❌ Web服务器模块加载失败: {e}")
        return 1
    except Exception as e:
        print(f"❌ Web服务器运行错误: {e}")
        return 1

def run_cli_recognition(image_path, method='fusion', output=None):
    """运行命令行识别"""
    # 验证输入文件
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return 1
    
    try:
        # 使用recognition pipeline直接进行识别
        from plate_recognition.pipeline.recognition_pipeline import RecognitionPipeline
        
        print(f"🔍 开始识别图片: {image_path}")
        print(f"🎯 使用方法: {method}")
        
        pipeline = RecognitionPipeline()
        result = pipeline.recognize_from_file(
            image_path=image_path,
            detection_method=method,
            return_debug_info=True
        )
        
        if result['success']:
            print(f"✅ 识别成功: {result['plate_number']}")
            print(f"📊 置信度: {result['confidence']:.2f}")
            print(f"⏱️  处理时间: {result['processing_time']:.3f}s")
            
            # 保存结果到文件
            if output:
                import json
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"💾 结果已保存到: {output}")
            
            return 0
        else:
            print(f"❌ 识别失败: {result.get('error', '未知错误')}")
            return 1
            
    except ImportError as e:
        print(f"❌ 识别模块加载失败: {e}")
        print("🔧 请确保车牌识别模块已正确安装")
        return 1
    except Exception as e:
        print(f"❌ CLI识别错误: {e}")
        return 1

def run_gui_interface():
    """运行GUI界面"""
    try:
        import tkinter as tk
        print("🖥️ 启动GUI界面...")
        print("⚠️  GUI模块尚未完全实现，请使用Web界面或CLI模式")
        print("💡 建议使用: python launcher.py web")
        return 0
        
    except ImportError as e:
        print(f"❌ GUI启动失败: {e}")
        print("请确保已安装tkinter等GUI依赖")
        return 1
    except Exception as e:
        print(f"❌ GUI运行错误: {e}")
        return 1

def run_api_server(host='0.0.0.0', port=8000):
    """运行纯API服务器"""
    try:
        from api.v1.endpoints import api_v1_bp
        from flask import Flask
        
        # 条件导入CORS
        try:
            from flask_cors import CORS
            cors_available = True
        except ImportError:
            cors_available = False
            print("⚠️  Warning: flask-cors不可用，CORS支持已禁用")
        
        app = Flask(__name__)
        if cors_available:
            CORS(app)
        app.register_blueprint(api_v1_bp, url_prefix='/api/v1')
        
        print(f"🔌 启动API服务器...")
        print(f"📍 API地址: http://{host}:{port}/api/v1")
        print(f"📖 健康检查: http://{host}:{port}/api/v1/health")
        
        app.run(host=host, port=port, debug=False)
        
    except ImportError as e:
        print(f"❌ API服务器启动失败: {e}")
        print("请确保已安装Flask等API依赖")
        return 1
    except Exception as e:
        print(f"❌ API服务器运行错误: {e}")
        return 1

def run_batch_processing(input_dir, output_dir=None, method='fusion'):
    """运行批量处理"""
    try:
        import os
        import json
        from pathlib import Path
        from plate_recognition.pipeline.recognition_pipeline import RecognitionPipeline
        
        print(f"📁 开始批量处理: {input_dir}")
        print(f"🎯 使用方法: {method}")
        
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ 输入目录不存在: {input_dir}")
            return 1
        
        # 获取图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.rglob('*') 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ 在目录 {input_dir} 中未找到图片文件")
            return 1
        
        print(f"📊 找到 {len(image_files)} 个图片文件")
        
        # 创建输出目录
        output_path = None
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        pipeline = RecognitionPipeline()
        results = []
        success_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 处理: {image_file.name}")
            
            try:
                result = pipeline.recognize_from_file(
                    image_path=str(image_file),
                    detection_method=method
                )
                
                result['file_path'] = str(image_file)
                results.append(result)
                
                if result['success']:
                    success_count += 1
                    print(f"  ✅ {result['plate_number']} (置信度: {result['confidence']:.2f})")
                else:
                    print(f"  ❌ 识别失败")
                    
            except Exception as e:
                print(f"  ❌ 处理错误: {e}")
                results.append({
                    'file_path': str(image_file),
                    'success': False,
                    'error': str(e)
                })
        
        # 保存结果
        if output_path:
            results_file = output_path / 'batch_results.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 批量处理结果已保存到: {results_file}")
        
        print(f"\n📊 批量处理完成: {success_count}/{len(image_files)} 成功")
        return 0 if success_count > 0 else 1
            
    except ImportError as e:
        print(f"❌ 批量处理模块加载失败: {e}")
        print("请确保车牌识别模块已正确安装")
        return 1
    except Exception as e:
        print(f"❌ 批量处理错误: {e}")
        return 1

def run_system_test():
    """运行系统测试"""
    try:
        from tests.test_suite import TestSuite
        
        print("🧪 开始系统测试...")
        test_suite = TestSuite()
        test_results = test_suite.run_comprehensive_test()
        
        # 生成测试报告
        report = test_suite.generate_test_report(test_results)
        print(report)
        
        # 根据测试结果返回状态码
        summary = test_results.get('summary', {})
        pass_rate = summary.get('overall_pass_rate', 0)
        
        if pass_rate >= 0.8:
            print(f"\n✅ 测试通过 (通过率: {pass_rate:.1%})")
            return 0
        else:
            print(f"\n❌ 测试失败 (通过率: {pass_rate:.1%})")
            return 1
        
    except ImportError as e:
        print(f"❌ 测试模块加载失败: {e}")
        return 1
    except Exception as e:
        print(f"❌ 系统测试错误: {e}")
        return 1

def run_system_monitor():
    """运行系统监控"""
    try:
        from monitoring.system_monitor import SystemMonitor
        
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        print("📊 系统监控已启动，按Ctrl+C停止...")
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\n✅ 系统监控已停止")
        
    except ImportError as e:
        print(f"❌ 监控模块加载失败: {e}")
        return 1
    except Exception as e:
        print(f"❌ 系统监控错误: {e}")
        return 1

def show_system_info():
    """显示系统信息"""
    try:
        from plate_recognition.core.config import get_config
        
        # 定义版本和作者信息
        VERSION = "2.0.0"
        AUTHOR = "zhen doniajohary2677@gmail.com"
        
        config = get_config()
        
        print("=" * 50)
        print("🚗 车牌识别系统信息")
        print("=" * 50)
        print(f"版本: {VERSION}")
        print(f"作者: {AUTHOR}")
        print(f"Python版本: {sys.version.split()[0]}")
        print(f"项目路径: {project_root}")
        print(f"配置文件: {getattr(config, 'config_path', '未指定')}")
        print("=" * 50)
        
        # 检查依赖
        print("📦 依赖检查:")
        dependencies = [
            ('opencv-python', 'cv2'),
            ('numpy', 'numpy'),
            ('pillow', 'PIL'),
            ('flask', 'flask'),
            ('paddleocr', 'paddleocr'),
            ('torch', 'torch'),
            ('tensorflow', 'tensorflow')
        ]
        
        for dep_name, import_name in dependencies:
            try:
                __import__(import_name)
                print(f"  ✅ {dep_name}")
            except ImportError:
                print(f"  ❌ {dep_name} (未安装)")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 获取系统信息失败: {e}")

def check_dependencies(required_deps):
    """检查依赖是否安装"""
    missing_deps = []
    for dep_name, import_name in required_deps:
        try:
            __import__(import_name)
        except ImportError:
            missing_deps.append(dep_name)
    return missing_deps

def print_dependency_help(missing_deps, context=""):
    """打印依赖安装帮助信息"""
    if missing_deps:
        print(f"🔧 缺少以下{context}依赖:")
        for dep in missing_deps:
            print(f"   ❌ {dep}")
        print("\n💡 安装命令:")
        print(f"   pip install {' '.join(missing_deps)}")
        print("   或者:")
        print("   pip install -r requirements.txt")

def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description='车牌识别系统 - 统一入口点',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式示例:
  %(prog)s web                    # 启动Web服务器
  %(prog)s api                    # 启动纯API服务器
  %(prog)s gui                    # 启动GUI界面
  %(prog)s cli image.jpg          # CLI单图识别
  %(prog)s batch input_dir        # 批量处理
  %(prog)s test                   # 运行系统测试
  %(prog)s monitor                # 启动系统监控
  %(prog)s info                   # 显示系统信息
        """
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # Web服务器模式
    web_parser = subparsers.add_parser('web', help='启动Web服务器')
    web_parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    web_parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    web_parser.add_argument('--debug', action='store_true', help='调试模式')
    
    # API服务器模式
    api_parser = subparsers.add_parser('api', help='启动API服务器')
    api_parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    api_parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    
    # GUI模式
    subparsers.add_parser('gui', help='启动GUI界面')
    
    # CLI模式
    cli_parser = subparsers.add_parser('cli', help='命令行识别')
    cli_parser.add_argument('image', help='图像文件路径')
    cli_parser.add_argument('--method', default='fusion', 
                           choices=['fusion', 'tesseract', 'paddleocr', 'crnn', 'gemini'],
                           help='识别方法')
    cli_parser.add_argument('--output', help='输出文件路径')
    
    # 批量处理模式
    batch_parser = subparsers.add_parser('batch', help='批量处理')
    batch_parser.add_argument('input_dir', help='输入目录')
    batch_parser.add_argument('--output', help='输出目录')
    batch_parser.add_argument('--method', default='fusion',
                             choices=['fusion', 'tesseract', 'paddleocr', 'crnn', 'gemini'],
                             help='识别方法')
    
    # 测试模式
    subparsers.add_parser('test', help='运行系统测试')
    
    # 监控模式
    subparsers.add_parser('monitor', help='启动系统监控')
    
    # 信息模式
    subparsers.add_parser('info', help='显示系统信息')
    
    # 全局参数
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默输出')
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(debug=args.verbose)
    
    # 如果没有指定模式，默认启动Web服务器
    if not args.mode:
        print("🚀 没有指定运行模式，默认启动Web服务器")
        print("💡 使用 --help 查看所有可用模式")
        return run_web_server()
    
    # 根据模式执行相应操作
    try:
        if args.mode == 'web':
            return run_web_server(args.host, args.port, args.debug)
        elif args.mode == 'api':
            return run_api_server(args.host, args.port)
        elif args.mode == 'gui':
            return run_gui_interface()
        elif args.mode == 'cli':
            return run_cli_recognition(args.image, args.method, args.output)
        elif args.mode == 'batch':
            return run_batch_processing(args.input_dir, args.output, args.method)
        elif args.mode == 'test':
            return run_system_test()
        elif args.mode == 'monitor':
            return run_system_monitor()
        elif args.mode == 'info':
            show_system_info()
            return 0
        else:
            print(f"❌ 未知的运行模式: {args.mode}")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序退出")
        return 0
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
